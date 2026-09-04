"""Deterministic synchronous federated-learning primitives.

Public objects are imported from their defining submodules. Keeping package
initialization side-effect free prevents a cycle between aggregation baselines and
the round server.
"""
