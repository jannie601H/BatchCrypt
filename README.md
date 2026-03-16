# BatchCrypt - Enhanced "BatchCrypt Zero-Skipping" Version

This repository contains an enhanced implementation of BatchCrypt based on the ATC'20 paper:  
**"BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning"**  
Original authors: [ATC'20 Paper](https://www.usenix.org/conference/atc20/presentation/ju)

> Enhanced by **Geonha Kim**, Hankuk University of Foreign Studies.

## Features

- Implements a novel **"BatchCrypt Zero-Skipping"** optimization that skips encryption of all-zero gradient blocks.
- Demonstrates significant reductions in training time with minimal impact on accuracy.
- Includes clipped quantization, batching, and homomorphic encryption based on the Paillier cryptosystem.

detail features in: BatchCrypt/paper/Contribution_Overview.pdf

## Recommended Setup

- Python 3.7+
- TensorFlow 2.x

## Related Publication

This repository contains the original implementation of the **BatchCrypt Zero-Skipping** method that served as the foundation for the following research paper:

**Optimizing Homomorphic Encryption in Federated Learning with Zero-Skipping**  
Accepted at [IEEE ICCE 2026]

Authors: Yoo-Bin Tae, Su-Jeong Park, Geon-Ha Kim, Seung-Ho Lim

### Contribution of this repository

The core implementation of **BatchCrypt Zero-Skipping** was developed by **Geonha Kim**, 
including:

- Designed and implemented the BatchCrypt Zero-Skipping method to eliminate unnecessary homomorphic encryption of zero-valued gradient blocks.
- Designed and implemented the aggregation procedure to correctly handle skipped blocks during encrypted federated learning aggregation.
- Integration of the zero-skipping mechanism into the BatchCrypt batching and encryption pipeline.

The accepted paper extends this implementation by introducing **skip-threshold–based experimentation** 
and additional empirical analysis.

paper in: BatchCrypt/paper/paper.pdf
