"""Repeats: unrolled or scanned (``repeat``), their passes' variables stacked into one
state (``stacked``), and pipelined over the mesh's stages (``pipelined``); and
rematerialization (``remat``). Both the scan and remat call modules as functions of
arrays alone (``pure``). Which branch and which pass a forward is tracing, which
precision and remat ask, is counted in one place (``tracing``)."""
