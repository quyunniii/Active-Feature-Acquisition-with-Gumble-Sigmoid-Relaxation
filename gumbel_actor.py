"""
Gumbel Actor-Critic for Longitudinal Feature Acquisition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import GumbelSigmoid, PlannerNet, MaskLayer
from utils import get_timestep_embedding


class GumbelActor(pl.LightningModule):
    """Gumbel Actor-Critic model for feature acquisition"""

    def __init__(self, predictor, num_time, num_feat, config, num_aux=0,
                 num_groups=None, group_to_feat_matrix=None,
                 feature_costs=None, aux_feature_costs=None):
        """
        Args:
            predictor: Trained prediction network (frozen)
            num_time: Number of timesteps
            num_feat: Number of features per timestep
            config: Configuration dict with hyperparameters
            num_aux: Number of auxiliary/static features (0 = no aux)
            num_groups: Acquirable units per timestep (defaults to num_feat)
            group_to_feat_matrix: (num_groups, num_feat) expansion matrix
            feature_costs: Per-group costs, length num_groups (default all 1s)
            aux_feature_costs: Per-aux-feature costs, length num_aux (default all 1s)
        """
        super().__init__()
        self.predictor = predictor
        self.num_time = num_time
        self.num_feat = num_feat
        self.config = config
        self.num_aux = num_aux
        self.num_groups = num_groups if num_groups is not None else num_feat

        self.save_hyperparameters(ignore=['predictor'])
        self.automatic_optimization = False

        # Freeze predictor
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad = False

        # Gumbel sigmoid for stochastic feature selection
        self.gumbel_sigmoid = GumbelSigmoid(tau=config['gate_tau'])

        # Mask layer (operates at feature level)
        self.mask_layer = MaskLayer(mask_size=num_time * num_feat, append=False)

        # Group-to-feature expansion matrix
        if group_to_feat_matrix is not None:
            self.register_buffer('group_to_feat', group_to_feat_matrix)
        else:
            self.register_buffer('group_to_feat', torch.eye(num_feat))

        # Per-group acquisition costs (length num_groups), tiled across timesteps
        if feature_costs is None:
            feature_costs = config.get('feature_costs', None)
        if feature_costs is not None:
            fc = torch.tensor(feature_costs, dtype=torch.float32)
        else:
            fc = torch.ones(self.num_groups, dtype=torch.float32)
        self.register_buffer('feature_costs', fc)
        self.register_buffer(
            'feature_costs_flat',
            fc.unsqueeze(0).expand(num_time, -1).reshape(-1),
        )

        # Per-aux-feature acquisition costs (length num_aux)
        if aux_feature_costs is None:
            aux_feature_costs = config.get('aux_feature_costs', None)
        if aux_feature_costs is not None and len(aux_feature_costs) > 0:
            afc = torch.tensor(aux_feature_costs, dtype=torch.float32)
        else:
            afc = torch.ones(max(num_aux, 1), dtype=torch.float32)
        if num_aux > 0 and len(afc) != num_aux:
            print(f"WARNING: aux_feature_costs length ({len(afc)}) != num_aux ({num_aux}), resizing with 1.0 padding")
            new_afc = torch.ones(num_aux, dtype=torch.float32)
            n = min(len(afc), num_aux)
            new_afc[:n] = afc[:n]
            afc = new_afc
        self.register_buffer('aux_feature_costs', afc)

        # Stage 1: auxiliary feature gate (fixed policy, same for all patients)
        if num_aux > 0:
            aux_init = config.get('aux_init_logit', 0.0)
            self.aux_logits = nn.Parameter(torch.full((num_aux,), aux_init))
            self.aux_cost_weight = config.get('aux_cost_weight', config['cost_weight'])
            if self.aux_cost_weight is None:
                self.aux_cost_weight = config['cost_weight']

        # Planner network
        time_emb_dim = config['time_emb_dim']
        # Input: x_masked(T*d) + mask(T*d) + time_emb + aux_acquired(num_aux) + aux_gates(num_aux)
        planner_input_dim = (num_time * num_feat) * 2 + time_emb_dim + num_aux * 2
        planner_out_dim = num_time * self.num_groups  # group-level gates

        self.planner_nn = PlannerNet(
            planner_input_dim,
            planner_out_dim,
            hidden=config['planner_hidden']
        )

        # Hyperparameters
        self.lr = config['lr']
        self.threshold = config['threshold']
        self.cost_weight = config['cost_weight']
        self.time_emb_dim = time_emb_dim

    def expand_group_gates_to_feat_mask(self, group_gates):
        """
        Expand group-level gates to feature-level mask.

        Args:
            group_gates: (B, T * num_groups)
        Returns:
            feat_mask: (B, T * num_feat)
        """
        B = group_gates.size(0)
        g = group_gates.reshape(B, self.num_time, self.num_groups)
        f = torch.matmul(g, self.group_to_feat)
        return f.reshape(B, self.num_time * self.num_feat)

    def feat_mask_to_group_mask(self, feat_mask):
        """
        Collapse feature-level mask to group-level.
        A group is acquired if ANY feature in it is acquired.

        Args:
            feat_mask: (B, T * num_feat)
        Returns:
            group_mask: (B, T * num_groups)
        """
        B = feat_mask.size(0)
        f = feat_mask.reshape(B, self.num_time, self.num_feat)
        g = torch.matmul(f, self.group_to_feat.T)
        g = (g > 0).float()
        return g.reshape(B, self.num_time * self.num_groups)

    def get_aux_gates(self, batch_size, mask_static=None):
        """apply Gumbel-Sigmoid to aux logits(baseline featrue gate)."""
        logits = self.aux_logits.unsqueeze(0).expand(batch_size, -1)
        gates = self.gumbel_sigmoid(logits, hard=True)
        if mask_static is not None:
            gates = gates * mask_static
        return gates

    def after_cur_t_mask(self, cur_t, T, d, device):
        """create mask for features at timesteps >= cur_t"""
        B = cur_t.size(0)
        t = torch.arange(T, device=device).view(1, T)
        allowed_t = t >= cur_t.view(B, 1)
        allowed = allowed_t.unsqueeze(-1).expand(B, T, d)
        return allowed.reshape(B, T * d)

    def predict_with_mask(self, x, mask, aux_acquired=None):
        """make predictions using the frozen predictor (feature-level mask)."""
        preds = []
        for t in range(self.num_time):
            m_t = mask.clone()
            m_t[:, (t + 1) * self.num_feat:] = 0

            x_t_masked = self.mask_layer(x, m_t)

            t_indicator = torch.full(
                (x.size(0),),
                (t + 1) / self.num_time,
                device=x.device
            ).unsqueeze(1)

            if aux_acquired is not None:
                x_t_masked = torch.cat((t_indicator, x_t_masked, aux_acquired), dim=1)
            else:
                x_t_masked = torch.cat((t_indicator, x_t_masked), dim=1)
            preds.append(self.predictor(x_t_masked))

        return torch.stack(preds, dim=1)

    def training_step(self, batch, batch_idx):
        """Training step"""
        opt = self.optimizers()

        #batch: tuple size 5 (no aux) or size 7 (with aux)
        if len(batch) == 7:
            x, y, mask, time_emb, cur_t, x_static, mask_static = batch
            x_static = x_static.to(self.device)
            mask_static = mask_static.to(self.device)
        else:
            x, y, mask, time_emb, cur_t = batch
            x_static = None
            mask_static = None

        # Reshape
        mask = mask.reshape(-1, self.num_time * self.num_feat)
        x = x.reshape(-1, self.num_time * self.num_feat)
        y = y.reshape(-1, self.num_time)

        x = x.to(self.device)
        mask = mask.to(self.device)
        cur_t = cur_t.to(self.device)
        time_emb = time_emb.to(self.device)
        y = y.to(self.device)

        B = x.size(0)
        orig_mask_f = mask.float()

        # Convert feature-level oracle mask to group-level
        orig_mask_g = self.feat_mask_to_group_mask(orig_mask_f)

        # auxillary gate
        aux_acquired = None
        aux_gates = None
        aux_cost_loss = 0.0
        if self.num_aux > 0 and x_static is not None:
            aux_gates = self.get_aux_gates(B, mask_static)
            aux_acquired = x_static * aux_gates
            aux_cost_loss = self.aux_cost_weight * (aux_gates * self.aux_feature_costs).sum() / B

        # group level availability mask
        mask_after_g = self.after_cur_t_mask(
            cur_t, self.num_time, self.num_groups, self.device
        ).float()

        # Forward through planner (input is feature-level)
        x_masked = self.mask_layer(x, mask)
        if aux_acquired is not None:
            planner_input = torch.cat([x_masked, mask, aux_acquired, aux_gates, time_emb], dim=1)
        else:
            planner_input = torch.cat([x_masked, mask, time_emb], dim=1)
        planner_logits = self.planner_nn(planner_input)  # (B, T*num_groups)

        #gumbel sigmoid at group level
        z_groups = self.gumbel_sigmoid(planner_logits, hard=True)

        #group acquisitions at allowed positions
        z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
        gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0.0, 1.0)

        #expand to feature level for prediction
        gated_mask_feat = self.expand_group_gates_to_feat_mask(gated_groups).clamp(0.0, 1.0)

        #predict with feature-level mask
        y_hat = self.predict_with_mask(x, gated_mask_feat, aux_acquired=aux_acquired)
        B, T, C = y_hat.shape

        #select labels for timesteps >= cur_t
        t_idx = torch.arange(T, device=y_hat.device).view(1, T)
        time_ok = (t_idx >= cur_t.view(B, 1))
        label_ok = (y != -1)
        use_for_CE = time_ok & label_ok

        y_hat_sel = y_hat[use_for_CE]
        y_sel = y[use_for_CE]

        CE_loss = F.cross_entropy(y_hat_sel, y_sel.long())

        #cost at group level (weighted by per-group costs)
        cost = (gated_groups * mask_after_g * self.feature_costs_flat).sum() / B
        cost_loss = self.cost_weight * cost

        loss = CE_loss + cost_loss + aux_cost_loss

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ce_loss', CE_loss, prog_bar=True)
        self.log('train/cost_loss', cost_loss, prog_bar=True)
        self.log('train/cost', cost, prog_bar=True)
        if self.num_aux > 0 and aux_gates is not None:
            self.log('train/aux_cost', aux_gates.sum() / B, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        if len(batch) == 7:
            x, y, mask, time_emb, cur_t, x_static, mask_static = batch
            x_static = x_static.to(self.device)
            mask_static = mask_static.to(self.device)
        else:
            x, y, mask, time_emb, cur_t = batch
            x_static = None
            mask_static = None

        mask = mask.reshape(-1, self.num_time * self.num_feat)
        x = x.reshape(-1, self.num_time * self.num_feat)
        y = y.reshape(-1, self.num_time)

        x = x.to(self.device)
        mask = mask.to(self.device)
        cur_t = cur_t.to(self.device)
        time_emb = time_emb.to(self.device)
        y = y.to(self.device)

        B = x.size(0)
        orig_mask_f = mask.float()
        orig_mask_g = self.feat_mask_to_group_mask(orig_mask_f)

        #aux gate
        aux_acquired = None
        aux_gates = None
        aux_cost_loss = 0.0
        if self.num_aux > 0 and x_static is not None:
            aux_gates = self.get_aux_gates(B, mask_static)
            aux_acquired = x_static * aux_gates
            aux_cost_loss = self.aux_cost_weight * (aux_gates * self.aux_feature_costs).sum() / B

        mask_after_g = self.after_cur_t_mask(
            cur_t, self.num_time, self.num_groups, self.device
        ).float()

        with torch.no_grad():
            x_masked = self.mask_layer(x, mask)
            if aux_acquired is not None:
                planner_input = torch.cat([x_masked, mask, aux_acquired, aux_gates, time_emb], dim=1)
            else:
                planner_input = torch.cat([x_masked, mask, time_emb], dim=1)
            planner_logits = self.planner_nn(planner_input)

            z_groups = self.gumbel_sigmoid(planner_logits, hard=True)

            z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
            gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0.0, 1.0)

            gated_mask_feat = self.expand_group_gates_to_feat_mask(gated_groups).clamp(0.0, 1.0)

            y_hat = self.predict_with_mask(x, gated_mask_feat, aux_acquired=aux_acquired)
            B, T, C = y_hat.shape

            t_idx = torch.arange(T, device=y_hat.device).view(1, T)
            time_ok = (t_idx >= cur_t.view(B, 1))
            label_ok = (y != -1)
            use_for_CE = time_ok & label_ok

            y_hat_sel = y_hat[use_for_CE]
            y_sel = y[use_for_CE]

            if len(y_sel) == 0:
                return None

            CE_loss = F.cross_entropy(y_hat_sel, y_sel.long())
            cost = (gated_groups * mask_after_g * self.feature_costs_flat).sum() / B
            cost_loss = self.cost_weight * cost
            loss = CE_loss + cost_loss + aux_cost_loss

        # Log validation metrics
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/ce_loss', CE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/cost_loss', cost_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/cost', cost, prog_bar=True, on_step=False, on_epoch=True)

        # Accuracy
        y_pred = torch.argmax(y_hat_sel, dim=-1)
        accuracy = (y_pred == y_sel).float().mean()
        self.log('val/accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)

        if self.num_aux > 0 and aux_gates is not None:
            self.log('val/aux_cost', aux_gates.sum() / B, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer"""
        params = list(self.planner_nn.parameters())
        if self.num_aux > 0:
            params.append(self.aux_logits)
        return torch.optim.Adam(params, lr=self.lr)
