"""Repeats: unrolled or scanned (``repeat``), their passes' variables stacked into one
state (``stacked``), and pipelined over the mesh's stages (``pipelined``); and
rematerialization (``remat``). Both the scan and remat call modules as functions of
arrays alone (``pure``). Which branch and which pass a forward is tracing, which
precision and remat ask, is counted in one place (``tracing``).

A ``.bux`` repeat, ``block^8``, is built by the run's ``methods.repeat``:
``methods(repeat=scan)`` stacks the passes and scans them, ``unroll`` builds each on
its own. ``methods.remat`` says what a residual branch outside any other, or a
scanned pass, keeps for the backward pass, and ``mesh.stage`` above 1 pipelines the
longest scanned repeat.
"""
