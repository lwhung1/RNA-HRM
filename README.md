# Hierarchical reasoning pipeline — RNA workflow & pseudocode

A focused, copyable implementation blueprint for applying **Hierarchical Reasoning Models (HRMs)** to RNA 3D structure prediction. Contains a workflow diagram, modular pseudocode, loss functions, MD-integration notes, and a practical checklist to get started.

---

# Workflow (Mermaid)

```mermaid
flowchart TD
  A[Input: Sequence + optional covariation/MSA / experimental constraints] --> B[Level 1 Sequence Encoding]
  B --> C[Level 2: Secondary structure (pairing map)]
  C --> D[Level 3: Motif extraction & tertiary contact prediction]
  D --> E[Level 4: Coarse 3D Layout (motif frames & stem orientations)]
  E --> F[Level 5: Atomistic refinement (equivariant GNN / diffusion)]
  F --> G[Physics-based refinement (ions, MD)]
  G --> H[Functional / ensemble reasoning & ligand binding]
  H --> I[Output: final 3D ensemble + per-residue uncertainty]

  subgraph Cross-scale
    B <--> C
    C <--> D
    D <--> E
    E <--> F
  end

  style Cross-scale stroke-dasharray: 5 5
```

---

# High-level description

1. **Sequence Encoding**: nucleotide embeddings (one-hot or pretrained RNA LM) and chemical modification flags.
2. **Secondary structure prediction**: neural + thermodynamic hybrid (probabilistic pairing map) — the central HRM prior.
3. **Motif extraction & tertiary contact prediction**: pool stems, junctions, loops into motif nodes; predict motif–motif contacts including pseudoknots.
4. **Coarse 3D layout**: predict motif centroids and local frames (position + orientation) or low-res backbone trace (sugar-phosphate trace).
5. **Atomistic refinement**: SE(3)-equivariant GNN or diffusion conditioned on coarse layout and pairing map to output nucleotide atomic coordinates.
6. **Physics-based refinement**: add ions (Mg²⁺/K⁺), energy-minimize and short MD to enforce sugar pucker and ion-mediated contacts.
7. **Functional/ensemble reasoning**: produce alternative states (ligand-free/bound, different ionation) plus uncertainty estimates.

---

# Pseudocode — overall pipeline (Python-like)

```python
class RNA_HRM(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Level 1 encoder
        self.seq_encoder = RNASeqEncoder(...)
        # Level 2: secondary predictor (probabilistic pairing matrix)
        self.secondary_net = SecondaryPredictor(...)
        # Level 3: motif GNN
        self.motif_gnn = MotifGNN(...)
        # Level 4: coarse layout predictor
        self.coarse_layout = CoarseLayoutNet(...)
        # Level 5: equivariant refiner
        self.refiner = EquivariantRefiner(...)
        # Cross-scale fusion
        self.cross_attn = CrossScaleAttention(...)

    def forward(self, seq, msa=None, constraints=None):
        seq_feats = self.seq_encoder(seq, msa)
        pair_probs = self.secondary_net(seq_feats, constraints)
        motif_graph = build_motif_graph(seq, pair_probs)
        motif_feats = self.motif_gnn(motif_graph)
        coarse = self.coarse_layout(motif_feats)
        fused = self.cross_attn(seq_feats, motif_feats, coarse)
        atom_coords = self.refiner(fused, coarse, pair_probs)
        return {
            'pair_probs': pair_probs,
            'motifs': motif_feats,
            'coarse': coarse,
            'coords': atom_coords,
        }
```

# Training loop sketch

```python
for batch in dataloader:
    outputs = model(batch['seq'], msa=batch.get('msa'), constraints=batch.get('constraints'))
    loss = 0
    # Secondary structure loss (cross-entropy on pair matrix / base-pair precision)
    loss += L_pair(outputs['pair_probs'], batch['pair_labels']) * w_pair
    # Coarse layout loss (frame / centroid alignment)
    loss += L_frame(outputs['coarse'], batch['coarse_targets']) * w_frame
    # Atom-level loss (FAPE or RMSD masked by base-planarity)
    loss += L_atom(outputs['coords'], batch['coords']) * w_atom
    # Geometric priors: base planarity, sugar pucker, bond lengths/angles
    loss += L_rna_geom(outputs['coords']) * w_geom
    # Tertiary contact loss, pseudoknot supervision
    loss += L_contact(outputs['motifs'], batch['ter_contacts']) * w_contact
    loss.backward()
    optimizer.step()
```

---

# Key RNA-specific components & losses

- **Pairing map loss**: supervise probabilistic NxN pairing matrix (symmetric); optionally include pseudoknot class labels.
- **Base-planarity loss**: each base's heavy atoms should lie close to a fitted plane; penalize out-of-plane deviation.
- **Sugar-pucker prior**: penalize improbable C3'-endo vs C2'-endo geometries depending on RNA type.
- **Glycosidic angle (chi) regularizer**: keep nucleotide chi within realistic ranges for A-form.
- **Stacking/coaxial stacking loss**: prefer coplanar, offset stacking for adjacent bases in stems.
- **Ion coordination loss**: when Mg²⁺ is predicted or known, enforce coordination distances to phosphate oxygens/waters.
- **Atom-level loss options**: Frame-Aligned Point Error (FAPE) is robust; use masked RMSD for local regions.

---

# Cross-scale reasoning patterns

- **Bottom-up:** strong base-pair probabilities produce motif nodes (stems/junctions) and constrain motif placement.
- **Top-down:** motif-level tertiary contacts and ion-binding predictions can flip ambiguous base-pair probabilities (resolve alt folds/pseudoknots).
- **Lateral:** within-level message passing between motifs to assemble global topology (e.g., coaxial stacking across junctions).

---

# Inference & sampling strategies

- **Deterministic inference:** single forward pass yields best-guess structure + confidence.
- **Stochastic sampling:** sample pairing matrices (Monte Carlo or diffusion) to get alternative secondary structures, then generate ensembles of 3D models.
- **Conditional sampling:** fix known stems or experimental constraints (SAXS, SHAPE, mutate data) and sample remainder.
- **Multi-state outputs:** produce multiple hypotheses ranked by predicted LDDT-like confidence.

---

# MD & refinement integration

1. Convert predicted model to a force-field consistent PDB (add hydrogens, set protonation if modified bases).
2. Place ions: K⁺ background, Mg²⁺ at predicted coordination sites; allow waters to mediate coordination.
3. Constrained minimization: restrain heavy atoms moderately while optimizing hydrogen positions and ion placements.
4. Short restrained MD (100–500 ps) then gradual release to sample local relaxation.
5. Optionally run enhanced-sampling (accelerated MD, metadynamics) to explore alternative conformations.
6. Use model confidence to determine per-region restraint strengths.

---

# Tools & libraries

- **Sequence & MSA:** custom RNA MSAs, RNA-MSM-like LMs
- **Secondary:** ViennaRNA, CONTRAfold, E2Efold
- **Motif libraries:** RNA3Dmotif, FR3D
- **Equivariant networks:** e3nn, SE3-Transformer, GVP
- **Diffusion:** RFdiffusion-style pipelines adapted for RNA
- **MD / refinement:** OpenMM, AMBER (RNA FF like OL3), GROMACS
- **Experimental integration:** SHAPE/ICSHAPE data, cryo-EM maps, SAXS

---

# Practical checklist to implement

- [ ] Curate dataset: non-redundant RNA PDBs, annotated secondary structures, motif labels.
- [ ] Preprocessing: extract base-pair maps, sugar pucker labels, motif centroids, and ion-binding sites.
- [ ] Implement Level 2 predictor first (secondary + pseudoknots), then motif extraction.
- [ ] Train coarse layout modules before atomistic refiner; this stabilizes learning.
- [ ] Add RNA-specific geometric regularizers early (base plane, sugar pucker).
- [ ] Integrate SHAPE/experimental data as optional constraints during inference.

---

# Example application: riboswitch aptamer with Mg²⁺-dependent folding

- Secondary prior predicts multiple possible stems.
- Motif module predicts an Mg²⁺-binding pocket coordinating three phosphate oxygens.
- Top-down signal from Mg²⁺ pocket promotes a particular pseudoknot, producing the closed aptamer state.
- Refiner outputs atomic model; MD with explicit Mg²⁺ confirms stable coordination and ligand-binding geometry.

---

# Quick references (terms to search)

- Pseudoknot-aware secondary prediction
- Frame-Aligned Point Error (FAPE) for RNA
- Sugar-pucker (C3'-endo vs C2'-endo)
- Base-planarity & stacking metrics
- Mg²⁺ coordination geometry in RNA


---

_End of document._

