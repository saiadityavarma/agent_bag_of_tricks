# Agent Bag of Tricks

This repository collects techniques and integrations designed to make cloud agent SDKs production ready. It distills ideas from recent state-of-the-art (SOTA) papers and brings together open-source frameworks that enhance the performance and reliability of autonomous agents.

## Goals

- Apply SOTA research to strengthen agent reasoning and planning.
- Integrate with popular open-source tools and frameworks.
- Share best practices that enable robust and scalable deployments.

## Getting Started

The repository will evolve as more features and integrations are added. Check back for updates and examples as development continues.

## Contributing

Contributions are welcome. Feel free to open issues or pull requests with improvements, new integrations, or references to emerging research.


## Unified Agent Platform

The `unified_platform` package provides a single interface for managing agents across Azure, Bedrock, and GCP SDKs. It exposes helper classes for creating agents, threads, functions, telemetry, and agent connections. Stub clients demonstrate the expected API for each provider.
