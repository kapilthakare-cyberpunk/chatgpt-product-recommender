# ChatGPT Product Recommender

This repository contains a complete product recommendation system with both web and CLI interfaces, leveraging AI models for generating product suggestions.

## Project Structure

```
chatgpt-product-recommender/
├── magento-product-item-recommendor-ai/  # Original web-based recommendation system
│   ├── backend/
│   ├── frontend/
│   ├── data/
│   └── ...
└── cli-tool/                            # New CLI tool with AI integration
    ├── ai_providers/
    ├── commands/
    ├── config/
    ├── docs/
    ├── tests/
    ├── utils/
    ├── chatgpt_product_recommender.py
    ├── setup.py
    ├── pyproject.toml
    ├── requirements.txt
    ├── README.md
    ├── Makefile
    ├── .gitignore
    ├── LICENSE
    ├── CHANGELOG.md
    ├── CONTRIBUTING.md
    ├── CODE_OF_CONDUCT.md
    └── .github/
        └── workflows/
```

## Components

### Magento Product Recommender AI (Original)
- A web-based product recommendation system
- Built with FastAPI backend and Next.js frontend
- Uses scikit-learn for similarity-based recommendations
- Connects to Magento price lists

### CLI Tool (New Addition)
- Command-line interface for product recommendations
- Supports multiple AI providers (OpenAI, Anthropic, Groq, Google Gemini, Ollama)
- Secure configuration management
- Comprehensive error handling
- Modular architecture

## Getting Started

For the CLI tool, see the [CLI README](cli-tool/README.md) for installation and usage instructions.

For the original web application, see the [Magento README](magento-product-item-recommendor-ai/README.md).

## License

MIT