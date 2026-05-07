"""Shared variational-inference primitives.

Math helpers used across multiple models — Newton steps for Dirichlet
concentration parameters, closed-form M-steps for Beta(1, β) stick
concentrations, and similar primitives that don't belong to any single
model. Models compose these; tests exercise them in isolation.
"""
