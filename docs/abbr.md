# Abbreviations

Status: active glossary. This file is a short naming aid only; the controlling
contract remains `docs/Glider_Control_Project_Plan.md`,
`docs/Daily_Schedule.txt`, `docs/Skills.md`,
`docs/Python Coding Instruction.txt`, `docs/MATLAB Coding.txt`, and
`docs/housekeeping_and_naming_rules.md`.

Use these abbreviations consistently in paths, reports, manifests, and compact
tables when a shorter label is helpful.

```text
ctx       environment context
prim      primitive
primvar   primitive-controller variant
ctrl      controller
gov       viability governor
sel       selector
mem       directional residual lift memory
ep        episode
rf        real flight
sr        sim-real replay
nom       nominal
rand      randomised
sum       summary
diag      diagnostic_not_passed / not move-on evidence

w0        dry-air / near-dry R5 synthesis layer
w1        randomized measured-updraft R5 synthesis layer
w2        optional GP-corrected annular-Gaussian fixed-LQR diagnostic layer
w3        frozen held-out randomised GP-corrected annular-Gaussian fixed-LQR validation layer
w01       combined R5 W0/W1 robust primitive-library synthesis
post_w3   post-W3 library-size cross-study and optional compression
flv       repeated-launch full-loop validation

q         LQR state-weight matrix
r         LQR input-weight matrix
k         LQR gain matrix
fifo      command-delay first-in-first-out timing state
roa       region of attraction
```

Do not use old R6/R7/R8/R9 labels, selected-controller labels, or
hardware-shortlist labels for new active evidence. Historical files may mention
those labels only with an explicit retired or diagnostic boundary.
