# BatchCrypt - Enhanced "Batch Zero" Version

This repository contains an enhanced implementation of BatchCrypt based on the ATC'20 paper:  
**"BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning"**  
Original authors: [ATC'20 Paper](https://www.usenix.org/conference/atc20/presentation/ju)

> Enhanced by **Kunha Kim**, Hankuk University of Foreign Studies.

## 🔧 Features

- Implements a novel **"Batch Zero"** optimization that skips encryption of all-zero gradient blocks.
- Demonstrates significant reductions in training time with minimal impact on accuracy.
- Includes quantization, stochastic rounding, and homomorphic encryption based on the Paillier cryptosystem.

## 🧪 Recommended Setup

- Python 3.7+
- TensorFlow 2.x
