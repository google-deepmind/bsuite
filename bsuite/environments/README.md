# Environments

This folder contains the raw *environments* used in `bsuite` experiments; we
expose them here for debugging and development purposes;

Recall that in the context of bsuite, an *experiment* consists of three parts:
1. Environments: a fixed set of environments determined by some parameters. 2.
Interaction: a fixed regime of agent/environment interaction (e.g. 100
episodes). 3. Analysis: a fixed procedure that maps agent behaviour to results
and plots.

Note: If you load the environment from this folder you will miss out on the
interaction+analysis as specified by bsuite. In general, you should use the
`bsuite_id` to load the environment via `bsuite.load_from_id(bsuite_id)` rather
than the raw environment.
