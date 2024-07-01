# Byte-Pair Encoding Tokenizer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Overview <a name="overview"></a>
This project introduces a robust and lightweight byte-pair encoding (BPE) tokenizer, designed to handle linguistic data efficiently. While it primarily focuses on the English language, its architecture allows it to model any language effectively. The tokenizer is equipped with a comprehensive vocabulary of approximately 15,000 tokens, demonstrating strong adaptability to a variety of unseen text inputs.

The tokenizer is compiled for web use, leveraging the power of WebAssembly through wasm-bindgen. This makes it readily accessible for integration into web applications, providing a seamless user experience without the need for server-side processing.

**Planned future enhancements include:**
- *GPU Implementation:* Using WGPU for processing larger training corpora and achieving faster training and tokenization.
- *Expanded Language Support:* Developing capabilities for capitalized and multilingual tokenization post-performance improvements.


## Installation <a name="installation"></a>
**To run this project, execute the following commands in your terminal:**
```sh
git clone https://github.com/krrud/rust-bpe.git
cd rust-bpe
cargo build
```


## Usage <a name="usage"></a>
**To train the tokenizer with your corpus:**
```rust
let corpus = "path/to/corpus/dir";
let output = "path/to/output-file.json";
let iterations = 15000;
let tokenizer = Tokenizer::train_cpu(corpus, iterations, Some(output));
```

**To tokenize text and retrieve tokens or indices:**
```rust
let tokenizer = Tokenizer::load("path/to/your/trained-tokenizer.json").unwrap();
let tokens = tokenizer.tokenize("your text to tokenize");
let token_vals = tokenizer.get_tokens(&tokens);
let token_indices = tokenizer.get_indices(&token_vals);
```

**To convert tokens back to their original string:**
```rust
let detokenized = tokenizer.detokenize(&tokens);
```


## Acknowledgements <a name="acknowledgements"></a>
Training data was graciously provided by [The Gutenberg Project](https://www.gutenberg.org/).


## License <a name="license"></a>
This project is licensed under the MIT License - see the LICENSE.md file for details.