"""The tabular learner and its memory.

What one visit banks (`regret_rules`), the four flat arrays every ledger is a
window into (`packed_table`, `infoset_table`), the sampled walk (`mccfr`),
W walks per seat against one frozen strategy (`lockstep`, the reference, and
`parallel`, the same across W processes, bit for bit), and a run that stops
and resumes (`lockstep_run`).
"""
