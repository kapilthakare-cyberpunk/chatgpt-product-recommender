#!/bin/bash

# Comprehensive script to recreate and push ChatGPT Product Recommender CLI to GitHub
# Assumes you have GitHub CLI installed and authenticated (`gh auth login`)

set -e  # Exit on any error

echo "🚀 Automating GitHub repository creation and push for ChatGPT Product Recommender"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI is not installed. Please install it first:"
    echo "  - On Ubuntu/Debian: sudo apt install gh"
    echo "  - On macOS: brew install gh"
    echo "  - On Windows: winget install GitHub.cli"
    echo ""
    echo "Then authenticate with: gh auth login"
    exit 1
fi

# Check if already authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Please run: gh auth login"
    exit 1
fi

# Get the username from gh config
USERNAME=$(gh api user --jq '.login')
echo "👤 Detected GitHub username: $USERNAME"

# Define repository names
CLI_REPO="chatgpt-product-recommender-cli"
MAIN_REPO="chatgpt-product-recommender"

# Define project directory
PROJECT_ROOT="/home/kapilt/Projects/pnz-projects/chatgpt-product-recommender"

# Check if project directory exists
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ Project directory does not exist: $PROJECT_ROOT"
    exit 1
fi

CLI_TOOL_DIR="$PROJECT_ROOT/cli-tool"

# Recreate the complete CLI tool directory structure
echo "🔧 Recreating CLI tool directory structure..."
mkdir -p "$CLI_TOOL_DIR"
mkdir -p "$CLI_TOOL_DIR/ai_providers"
mkdir -p "$CLI_TOOL_DIR/commands"
mkdir -p "$CLI_TOOL_DIR/config"
mkdir -p "$CLI_TOOL_DIR/docs"
mkdir -p "$CLI_TOOL_DIR/tests"
mkdir -p "$CLI_TOOL_DIR/utils"
mkdir -p "$CLI_TOOL_DIR/.github/workflows"

# Initialize git repo for CLI tool
cd "$CLI_TOOL_DIR"
git init
git config user.name "Kapil"
git config user.email "kapil@example.com"

# Create chatgpt_product_recommender.py file
cat > chatgpt_product_recommender.py << 'EOF'
#!/usr/bin/env python3
"""
ChatGPT Product Recommender CLI Tool

A command-line tool for generating product recommendations using AI models.
Supports multiple AI providers: OpenAI, Anthropic, Groq, Google Gemini, and Ollama.
"""
import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from commands.recommend import recommend_command
from commands.configure import configure_command
from commands.list_models import list_models_command
from utils.logger import logger


def main():
    logger.info("Starting ChatGPT Product Recommender CLI")
    
    parser = argparse.ArgumentParser(
        prog="chatgpt-product-recommender",
        description="AI-Powered Product Recommender CLI",
        epilog="Use 'chatgpt-product-recommender <command> --help' for more information on a specific command"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate product recommendations')
    recommend_parser.add_argument('--product', '-p', required=True, help='Product description or ID to base recommendations on')
    recommend_parser.add_argument('--count', '-c', type=int, default=5, help='Number of recommendations to generate (default: 5)')
    recommend_parser.add_argument('--provider', '-prov', choices=['openai', 'anthropic', 'groq', 'gemini', 'ollama'], 
                                  help='AI provider to use (default: will use configured default)')
    recommend_parser.add_argument('--model', '-m', help='Specific model to use (provider dependent)')
    recommend_parser.set_defaults(func=recommend_command)
    
    # Configure command
    configure_parser = subparsers.add_parser('configure', help='Configure AI provider settings')
    configure_parser.add_argument('--provider', '-prov', choices=['openai', 'anthropic', 'groq', 'gemini', 'ollama'], 
                                  required=True, help='AI provider to configure')
    configure_parser.add_argument('--api-key', '-key', help='API key for the provider (optional, will prompt if not provided)')
    configure_parser.add_argument('--model', '-m', help='Default model to use for this provider')
    configure_parser.set_defaults(func=configure_command)
    
    # List models command
    list_models_parser = subparsers.add_parser('list-models', help='List available models for a provider')
    list_models_parser.add_argument('--provider', '-prov', choices=['openai', 'anthropic', 'groq', 'gemini', 'ollama'], 
                                    required=True, help='AI provider to list models for')
    list_models_parser.set_defaults(func=list_models_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is provided, show help
    if args.command is None:
        parser.print_help()
        logger.info("Displayed help information")
        sys.exit(1)
    
    # Execute the appropriate function
    try:
        logger.info(f"Executing command: {args.command}")
        args.func(args)
        logger.info(f"Command {args.command} completed successfully")
    except KeyboardInterrupt:
        print("\n🚫 Operation cancelled by user.")
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"🚫 Error: {str(e)}", file=sys.stderr)
        logger.error(f"Uncaught exception in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
EOF

# Create ai_providers/__init__.py
cat > ai_providers/__init__.py << 'EOF'
"""
AI Provider interface for the ChatGPT Product Recommender CLI.
Abstract base class and implementations for different AI providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import config_manager


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.api_key = config_manager.get_api_key(provider_name)
    
    @abstractmethod
    def get_completion(self, prompt: str, model: Optional[str] = None) -> str:
        """Get a completion from the AI model."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""
    
    def __init__(self):
        super().__init__("openai")
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Use 'chatgpt-product-recommender configure --provider openai' to set it.")
        
        self.client = self.OpenAI(api_key=self.api_key)
    
    def get_completion(self, prompt: str, model: Optional[str] = "gpt-3.5-turbo") -> str:
        if not model:
            model = "gpt-3.5-turbo"  # default model
            
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests products based on user descriptions. Be concise and specific with your recommendations. Return only the product recommendations in a clear format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def list_models(self) -> List[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]


class AnthropicProvider(AIProvider):
    """Anthropic provider implementation."""
    
    def __init__(self):
        super().__init__("anthropic")
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("Please install the anthropic package: pip install anthropic")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not configured. Use 'chatgpt-product-recommender configure --provider anthropic' to set it.")
        
        self.client = self.anthropic.Anthropic(api_key=self.api_key)
    
    def get_completion(self, prompt: str, model: Optional[str] = "claude-3-haiku-20240307") -> str:
        if not model:
            model = "claude-3-haiku-20240307"  # default model
            
        response = self.client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.7,
            system="You are a helpful assistant that suggests products based on user descriptions. Be concise and specific with your recommendations. Return only the product recommendations in a clear format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def list_models(self) -> List[str]:
        # Anthropic doesn't provide an API to list models, so we return common ones
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307"
        ]


class GroqProvider(AIProvider):
    """Groq provider implementation."""
    
    def __init__(self):
        super().__init__("groq")
        try:
            from groq import Groq
            self.Groq = Groq
        except ImportError:
            raise ImportError("Please install the groq package: pip install groq")
        
        if not self.api_key:
            raise ValueError("Groq API key not configured. Use 'chatgpt-product-recommender configure --provider groq' to set it.")
        
        self.client = self.Groq(api_key=self.api_key)
    
    def get_completion(self, prompt: str, model: Optional[str] = "llama3-70b-8192") -> str:
        if not model:
            model = "llama3-70b-8192"  # default model
            
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests products based on user descriptions. Be concise and specific with your recommendations. Return only the product recommendations in a clear format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def list_models(self) -> List[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]


class GeminiProvider(AIProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self):
        super().__init__("gemini")
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("Please install the google-generativeai package: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("Gemini API key not configured. Use 'chatgpt-product-recommender configure --provider gemini' to set it.")
        
        self.genai.configure(api_key=self.api_key)
        self.model = None
    
    def get_completion(self, prompt: str, model: Optional[str] = "gemini-pro") -> str:
        if not model:
            model = "gemini-pro"  # default model
            
        if not self.model or self.model.model_name != f"models/{model}":
            self.model = self.genai.GenerativeModel(model)
            
        response = self.model.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )
        return response.text
    
    def list_models(self) -> List[str]:
        # List available models from the API
        models = self.genai.list_models()
        return [model.name.replace("models/", "") for model in models]


class OllamaProvider(AIProvider):
    """Ollama provider implementation."""
    
    def __init__(self):
        super().__init__("ollama")
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError("Please install the ollama package: pip install ollama")
    
    def get_completion(self, prompt: str, model: Optional[str] = "llama3") -> str:
        if not model:
            model = "llama3"  # default model
            
        response = self.ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests products based on user descriptions. Be concise and specific with your recommendations. Return only the product recommendations in a clear format."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 500
            }
        )
        return response['message']['content']
    
    def list_models(self) -> List[str]:
        models_response = self.ollama.list()
        return [model['name'] for model in models_response['models']]


def get_provider(provider_name: str) -> AIProvider:
    """Factory function to get the appropriate provider instance."""
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'groq': GroqProvider,
        'gemini': GeminiProvider,
        'ollama': OllamaProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {list(providers.keys())}")
    
    return providers[provider_name]()
EOF

# Create commands/__init__.py
cat > commands/__init__.py << 'EOF'
"""
Commands package for the ChatGPT Product Recommender CLI.
Contains all the command implementations.
"""
EOF

# Create commands/recommend.py
cat > commands/recommend.py << 'EOF'
"""
Recommend command for the ChatGPT Product Recommender CLI.
Handles product recommendation requests using AI models.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_providers import get_provider
from config.config_manager import config_manager
from utils.logger import logger


def recommend_command(args):
    """Execute the recommend command."""
    logger.info(f"Starting recommendation request for product: {args.product[:50]}...")
    
    # Determine which provider to use
    provider_name = args.provider or config_manager.get_default_provider()
    
    if not provider_name:
        print("❌ No provider specified and no default provider configured.")
        print("💡 Please configure a default provider or specify one with --provider.")
        print("💡 Use 'chatgpt-product-recommender configure --provider <provider>' to set up a provider.")
        return
    
    # Get the provider instance
    try:
        provider = get_provider(provider_name)
        logger.info(f"Successfully initialized {provider_name} provider")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except ImportError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Install the required package for {provider_name} provider.")
        return
    
    # Determine which model to use
    model = args.model
    
    # Create the prompt for product recommendations
    prompt = f"""
    Based on the following product, suggest {args.count} related products that a customer might be interested in:

    {args.product}

    Please provide the recommendations in the following format:
    1. Product Name: Brief description
    2. Product Name: Brief description
    3. Product Name: Brief description
    ...
    
    Focus on products that are similar in function, category, or complement the original product.
    """
    
    print(f"🚀 Generating {args.count} product recommendations using {provider_name}...")
    if model:
        print(f"🧠 Model: {model}")
    
    try:
        # Get recommendations from the AI model
        print("⏳ Processing your request...")
        recommendations = provider.get_completion(prompt, model)
        
        print("\n" + "🎨" + "="*48 + "🎨")
        print("💡 PRODUCT RECOMMENDATIONS")
        print("🎨" + "="*48 + "🎨")
        print(recommendations)
        print("🎨" + "="*48 + "🎨")
        
        logger.info("Recommendation request completed successfully")
        
    except Exception as e:
        print(f"🚫 Error generating recommendations: {e}")
        logger.error(f"Error in recommend_command: {e}")
EOF

# Create commands/configure.py
cat > commands/configure.py << 'EOF'
"""
Configure command for the ChatGPT Product Recommender CLI.
Handles configuration of AI provider settings.
"""
import getpass
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import config_manager
from utils.logger import logger


def configure_command(args):
    """Execute the configure command."""
    provider = args.provider
    logger.info(f"Starting configuration for provider: {provider}")
    
    # If API key not provided as argument, prompt for it
    if not args.api_key:
        if provider in ['ollama']:
            # Ollama doesn't require an API key
            api_key = ""
            print(f"ℹ️  {provider.capitalize()} doesn't require an API key.")
        else:
            api_key = getpass.getpass(f"🔐 Enter API key for {provider}: ")
    else:
        api_key = args.api_key
    
    try:
        # Get current provider config
        provider_config = config_manager.get_provider_config(provider)
        
        # Update the API key
        if api_key:
            provider_config['api_key'] = api_key
        
        # Update the model if provided
        if args.model:
            provider_config['model'] = args.model
        
        # Save the updated config
        config_manager.set_provider_config(provider, provider_config)
        
        # If this is the first provider being configured, set it as default
        if not config_manager.get_default_provider():
            config_manager.set_default_provider(provider)
            print(f"✅ Configuration saved! {provider} has been set as the default provider.")
        else:
            print(f"✅ Configuration saved for {provider}.")
        
        # Show what was configured
        print(f"  🔐 API Key: {'*' * len(provider_config.get('api_key', '')) if provider_config.get('api_key') else '(not set)'}")
        if 'model' in provider_config:
            print(f"  🧠 Default Model: {provider_config['model']}")
        
        print(f"\n✨ You can now use the {provider} provider in your recommendations.")
        logger.info(f"Configuration completed successfully for {provider}")
        
    except Exception as e:
        print(f"🚫 Error during configuration: {e}")
        logger.error(f"Error in configure_command: {e}")
EOF

# Create commands/list_models.py
cat > commands/list_models.py << 'EOF'
"""
List models command for the ChatGPT Product Recommender CLI.
Lists available models for a given AI provider.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_providers import get_provider
from utils.logger import logger


def list_models_command(args):
    """Execute the list-models command."""
    provider_name = args.provider
    logger.info(f"Requesting model list for provider: {provider_name}")
    
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except ImportError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Install the required package for {provider_name} provider.")
        return
    
    print(f"🔍 Available models for {provider_name}:")
    try:
        models = provider.list_models()
        if models:
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print("  🚫 No models found. The provider may not support listing models.")
        
        logger.info(f"Model listing completed for {provider_name}, found {len(models)} models")
    except Exception as e:
        print(f"🚫 Error listing models: {e}")
        logger.error(f"Error in list_models_command: {e}")
EOF

# Create config/config_manager.py
cat > config/config_manager.py << 'EOF'
"""
Configuration management for the ChatGPT Product Recommender CLI.
Handles API keys, provider settings, and default configurations.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional


class ConfigManager:
    def __init__(self):
        # Use home directory for config storage
        self.config_dir = Path.home() / '.chatgpt-product-recommender'
        self.config_file = self.config_dir / 'config.json'
        self.ensure_config_dir()
        
    def ensure_config_dir(self):
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def get_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_provider_config(self, provider: str) -> Dict:
        """Get configuration for a specific provider."""
        config = self.get_config()
        return config.get(provider, {})
    
    def set_provider_config(self, provider: str, settings: Dict):
        """Set configuration for a specific provider."""
        config = self.get_config()
        config[provider] = settings
        self.save_config(config)
    
    def get_default_provider(self) -> Optional[str]:
        """Get the default provider."""
        config = self.get_config()
        return config.get('default_provider')
    
    def set_default_provider(self, provider: str):
        """Set the default provider."""
        config = self.get_config()
        config['default_provider'] = provider
        self.save_config(config)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get the API key for a specific provider."""
        provider_config = self.get_provider_config(provider)
        return provider_config.get('api_key')
    
    def set_api_key(self, provider: str, api_key: str):
        """Set the API key for a specific provider."""
        provider_config = self.get_provider_config(provider)
        provider_config['api_key'] = api_key
        self.set_provider_config(provider, provider_config)


# Global config manager instance
config_manager = ConfigManager()
EOF

# Create utils/__init__.py
cat > utils/__init__.py << 'EOF'
"""
Utilities package for the ChatGPT Product Recommender CLI.
Contains helper functions and utilities.
"""
EOF

# Create utils/logger.py
cat > utils/logger.py << 'EOF'
"""
Logger utility for the ChatGPT Product Recommender CLI.
Provides consistent logging across the application.
"""
import logging
import sys
from pathlib import Path


def setup_logger():
    """Set up the application logger."""
    logger = logging.getLogger('chatgpt_product_recommender')
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optionally, add a file handler
    log_dir = Path.home() / '.chatgpt-product-recommender' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / 'app.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create a global logger instance
logger = setup_logger()
EOF

# Create tests/__init__.py
cat > tests/__init__.py << 'EOF'
"""
General tests and utilities for the ChatGPT Product Recommender CLI tool.
"""
import unittest


def run_all_tests():
    """Discover and run all tests."""
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    run_all_tests()
EOF

# Create tests/test_config.py
cat > tests/test_config.py << 'EOF'
"""
Test suite for the ChatGPT Product Recommender CLI tool.
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to the path so we can import our modules
import sys
sys.path.insert(0, str(str(Path(__file__).parent.parent)))

from config.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_config_dir = ConfigManager.config_dir
        ConfigManager.config_dir = self.test_dir
        self.config_manager = ConfigManager()
        # Override the config file path to use our test directory
        self.config_manager.config_file = self.test_dir / 'config.json'
    
    def tearDown(self):
        # Restore original config directory
        ConfigManager.config_dir = self.original_config_dir
        
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_ensure_config_dir(self):
        """Test that config directory is created."""
        config_manager = ConfigManager()
        config_manager.ensure_config_dir()
        self.assertTrue(config_manager.config_dir.exists())
    
    def test_save_and_get_config(self):
        """Test saving and retrieving configuration."""
        test_config = {'test_provider': {'api_key': 'test_key', 'model': 'test_model'}}
        self.config_manager.save_config(test_config)
        
        retrieved_config = self.config_manager.get_config()
        self.assertEqual(retrieved_config, test_config)
    
    def test_provider_config(self):
        """Test getting and setting provider configuration."""
        provider_config = {'api_key': 'test_key', 'model': 'test_model'}
        self.config_manager.set_provider_config('test_provider', provider_config)
        
        retrieved_config = self.config_manager.get_provider_config('test_provider')
        self.assertEqual(retrieved_config, provider_config)
    
    def test_default_provider(self):
        """Test setting and getting default provider."""
        self.config_manager.set_default_provider('test_provider')
        default_provider = self.config_manager.get_default_provider()
        self.assertEqual(default_provider, 'test_provider')
    
    def test_api_key(self):
        """Test setting and getting API key."""
        self.config_manager.set_api_key('test_provider', 'test_api_key')
        api_key = self.config_manager.get_api_key('test_provider')
        self.assertEqual(api_key, 'test_api_key')


if __name__ == '__main__':
    unittest.main()
EOF

# Create documentation files
cat > docs/architecture.md << 'EOF'
# Architecture Documentation

## Overview

The ChatGPT Product Recommender CLI is a command-line tool that generates product recommendations using multiple AI providers. The architecture follows a modular design with clear separation of concerns.

## Components

### 1. Main Application (`chatgpt_product_recommender.py`)
- Entry point for the CLI application
- Sets up argument parsing
- Routes commands to appropriate handlers

### 2. Commands (`commands/` directory)
- `recommend.py`: Handles product recommendation requests
- `configure.py`: Handles provider configuration
- `list_models.py`: Lists available models for providers

### 3. AI Providers (`ai_providers/` directory)
- Abstract base class for all AI providers
- Concrete implementations for each supported provider:
  - OpenAIProvider
  - AnthropicProvider
  - GroqProvider
  - GeminiProvider
  - OllamaProvider
- Factory function for creating provider instances

### 4. Configuration (`config/` directory)
- `config_manager.py`: Handles API keys and provider settings
- Stores configuration in user's home directory

### 5. Utilities (`utils/` directory)
- `logger.py`: Provides consistent logging across the application
- Additional utility functions

### 6. Tests (`tests/` directory)
- Unit tests for all components
- Mock implementations for testing without actual API calls

## Design Patterns

### Abstract Factory Pattern
The AI provider system uses an abstract factory pattern where providers inherit from a common `AIProvider` base class and are created through a factory function.

### Configuration Management
The configuration system uses a singleton pattern through the global `config_manager` instance.

## Data Flow

1. User runs a command with arguments
2. Main application parses arguments and routes to appropriate command handler
3. Command handler retrieves configuration and provider information
4. Provider generates response using AI model
5. Response is formatted and displayed to user
6. Errors are caught and appropriate messages displayed
EOF

cat > docs/api_reference.md << 'EOF'
# API Reference

## Main Functions

### `main()`
The main entry point for the CLI application. Sets up argument parsing and routes commands.

---

## Command Functions

### `recommend_command(args)`
Generate product recommendations based on user input.

**Parameters:**
- `args` (argparse.Namespace): Arguments containing product, count, provider, and model

### `configure_command(args)`
Configure AI provider settings.

**Parameters:**
- `args` (argparse.Namespace): Arguments containing provider, api_key, and model

### `list_models_command(args)`
List available models for a specific provider.

**Parameters:**
- `args` (argparse.Namespace): Arguments containing provider

---

## AI Provider Classes

### `AIProvider(ABC)`
Abstract base class for all AI providers.

#### Methods:
- `get_completion(self, prompt: str, model: Optional[str] = None) -> str`: Get a completion from the AI model
- `list_models(self) -> List[str]`: List available models for this provider

### `OpenAIProvider`
Implementation for OpenAI's API.

### `AnthropicProvider`
Implementation for Anthropic's API.

### `GroqProvider`
Implementation for Groq's API.

### `GeminiProvider`
Implementation for Google's Gemini API.

### `OllamaProvider`
Implementation for Ollama's local models API.

#### Methods:
- `get_provider(provider_name: str) -> AIProvider`: Factory function to get appropriate provider instance

---

## Configuration Manager

### `ConfigManager`
Manages configuration for the application.

#### Methods:
- `get_config(self) -> Dict`: Load configuration from file
- `save_config(self, config: Dict)`: Save configuration to file
- `get_provider_config(self, provider: str) -> Dict`: Get configuration for a specific provider
- `set_provider_config(self, provider: str, settings: Dict)`: Set configuration for a specific provider
- `get_default_provider(self) -> Optional[str]`: Get the default provider
- `set_default_provider(self, provider: str)`: Set the default provider
- `get_api_key(self, provider: str) -> Optional[str]`: Get the API key for a specific provider
- `set_api_key(self, provider: str, api_key: str)`: Set the API key for a specific provider
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
openai>=1.0.0
anthropic>=0.5.0
groq>=0.4.0
google-generativeai>=0.3.0
ollama>=0.1.0
pandas>=1.5.0
scikit-learn>=1.3.0

# Testing dependencies
pytest>=7.0.0
pytest-mock>=3.0.0
EOF

# Create setup.py
cat > setup.py << 'EOF'
"""
Setup configuration for the ChatGPT Product Recommender CLI tool.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chatgpt-product-recommender",
    version="0.1.0",
    author="Kapil",
    author_email="kapil@example.com",
    description="A CLI tool for generating product recommendations using AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chatgpt-product-recommender",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "chatgpt-product-recommender=chatgpt_product_recommender:main",
        ],
    },
)
EOF

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "chatgpt-product-recommender"
version = "0.1.0"
description = "A CLI tool for generating product recommendations using AI models"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Kapil", email = "kapil@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.8"
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.5.0",
    "groq>=0.4.0",
    "google-generativeai>=0.3.0",
    "ollama>=0.1.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-mock>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
]
test = [
    "pytest>=7.0.0",
    "pytest-mock>=3.0.0",
]

[project.scripts]
chatgpt-product-recommender = "chatgpt_product_recommender:main"

[project.urls]
Homepage = "https://github.com/yourusername/chatgpt-product-recommender"
Repository = "https://github.com/yourusername/chatgpt-product-recommender"
"Bug Tracker" = "https://github.com/yourusername/chatgpt-product-recommender/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["cli_tool*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v"

[tool.black]
line-length = 88
target-version = ['py38']
EOF

# Create README.md
cat > README.md << 'EOF'
# ChatGPT Product Recommender CLI

A command-line tool for generating product recommendations using AI models. Supports multiple AI providers including OpenAI, Anthropic, Groq, Google Gemini, and Ollama.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Commands](#commands)
- [AI Providers](#ai-providers)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Features

- 🤖 Support for multiple AI providers (OpenAI, Anthropic, Groq, Google Gemini, Ollama)
- 🔐 Secure API key management
- ⚙️ Flexible model selection per provider
- 📊 Product recommendation generation
- 🛠️ Easy-to-use command-line interface
- 📝 Comprehensive error handling and logging

## Installation

### Prerequisites
- Python 3.8 or higher
- API keys for the AI providers you want to use (except Ollama)

### Install from PyPI
```bash
pip install chatgpt-product-recommender-cli
```

### Install from Source
```bash
git clone https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli.git
cd chatgpt-product-recommender-cli
pip install -r requirements.txt
pip install -e .
```

## Configuration

Before using the tool, configure at least one AI provider. You can configure multiple providers and switch between them as needed.

### OpenAI
```bash
chatgpt-product-recommender-cli configure --provider openai --api-key <your-api-key>
```

### Anthropic
```bash
chatgpt-product-recommender-cli configure --provider anthropic --api-key <your-api-key>
```

### Groq
```bash
chatgpt-product-recommender-cli configure --provider groq --api-key <your-api-key>
```

### Google Gemini
```bash
chatgpt-product-recommender-cli configure --provider gemini --api-key <your-api-key>
```

### Ollama
```bash
chatgpt-product-recommender-cli configure --provider ollama
```

> **Note**: Ollama doesn't require an API key, but you need to have Ollama installed and running locally.

## Usage

### Generate Product Recommendations

Generate recommendations using the default provider:
```bash
chatgpt-product-recommender-cli recommend --product "wireless headphones with noise cancellation"
```

Specify the number of recommendations:
```bash
chatgpt-product-recommender-cli recommend --product "wireless headphones" --count 3
```

Use a specific provider:
```bash
chatgpt-product-recommender-cli recommend --product "laptop stand" --provider groq
```

Use a specific model:
```bash
chatgpt-product-recommender-cli recommend --product "smartphone" --provider openai --model gpt-4
```

### List Available Models

List models available for a specific provider:
```bash
chatgpt-product-recommender-cli list-models --provider openai
```

### Configuration Management

Set a default model for a provider:
```bash
chatgpt-product-recommender-cli configure --provider openai --model gpt-4
```

## Commands

### `recommend`
Generate product recommendations based on a product description.

**Options:**
- `--product, -p`: Product description or ID to base recommendations on (required)
- `--count, -c`: Number of recommendations to generate (default: 5)
- `--provider, -prov`: AI provider to use (default: configured default)
- `--model, -m`: Specific model to use (provider dependent)

### `configure`
Configure AI provider settings (API keys, default models).

**Options:**
- `--provider, -prov`: AI provider to configure (required)
- `--api-key, -key`: API key for the provider (will prompt if not provided)
- `--model, -m`: Default model to use for this provider

### `list-models`
List available models for a specific provider.

**Options:**
- `--provider, -prov`: AI provider to list models for (required)

## AI Providers

The tool supports multiple AI providers, each with their own strengths:

| Provider | Strengths | Common Models |
|----------|-----------|---------------|
| **OpenAI** | Advanced reasoning, broad knowledge | gpt-3.5-turbo, gpt-4, gpt-4-turbo |
| **Anthropic** | Helpful, honest, harmless responses | claude-3-sonnet, claude-3-opus, claude-3-haiku |
| **Groq** | High-speed inference | llama3-70b, llama3-8b, mixtral, gemma |
| **Google Gemini** | Multimodal capabilities | gemini-pro, gemini-1.5-pro |
| **Ollama** | Local inference, privacy | llama3, mistral, dolphin-mixtral |

## Examples

### Basic Recommendation
```bash
chatgpt-product-recommender-cli recommend --product "professional DSLR camera for portrait photography"
```

### Specific Provider and Model
```bash
chatgpt-product-recommender-cli recommend --product "mechanical keyboard for programming" --provider anthropic --model claude-3-sonnet
```

### Multiple Recommendations
```bash
chatgpt-product-recommender-cli recommend --product "wireless earbuds for running" --count 7
```

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Running from Source
```bash
python chatgpt_product_recommender.py --help
```

## License

MIT
EOF

# Create CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of ChatGPT Product Recommender CLI
- Support for multiple AI providers (OpenAI, Anthropic, Groq, Google Gemini, Ollama)
- Product recommendation functionality via AI models
- Command-line interface with recommend, configure, and list-models commands
- Configuration management for API keys and provider settings
- Comprehensive error handling and user feedback
- Logging functionality
- Unit tests for core components
- Documentation for installation, configuration, and usage
EOF

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Kapil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create Makefile
cat > Makefile << 'EOF'
# Makefile for ChatGPT Product Recommender CLI

.PHONY: install test lint format clean help

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@echo
	@grep -E '^[a-zA-Z_0-9%-]+:.*?## .*$$' $(word 1,$(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-deps: ## Install dependencies only
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run all tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=chatgpt_product_recommender --cov-report=html

lint: ## Lint the code
	flake8 chatgpt_product_recommender.py commands/ ai_providers/ config/ utils/
	black --check chatgpt_product_recommender.py commands/ ai_providers/ config/ utils/

format: ## Format the code
	black chatgpt_product_recommender.py commands/ ai_providers/ config/ utils/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

build: ## Build the package
	python -m build

publish: ## Build and publish to PyPI (requires twine)
	python -m build
	twine upload dist/*

dev-server: ## Run in development mode
	python chatgpt_product_recommender.py --help
EOF

# Create CONTRIBUTING.md
cat > CONTRIBUTING.md << 'EOF'
# Contributing to ChatGPT Product Recommender CLI

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
- [Styleguides](#styleguides)
- [Join The Project Team](#join-the-project-team)

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to <kapil@example.com>.

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](README.md).

Before you ask a question, it is best to search for existing [Issues](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](README.md). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to <kapil@example.com>.

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as `critical`), and the issue will be left to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for ChatGPT Product Recommender CLI, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](README.md) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/kapilthakare-cyberpunk/chatgpt-product-recommender-cli/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux. 
- **Explain why this enhancement would be useful** to most ChatGPT Product Recommender CLI users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

### Your First Code Contribution

1. Fork the repository
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/your-bug-fix`
3. Make your changes
4. Add tests if applicable
5. Run the tests to ensure everything works: `make test`
6. Commit your changes using a descriptive commit message
7. Push your branch to your fork
8. Create a pull request to the main repository

## Styleguides

### Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Follow the existing code style in the project

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## Join The Project Team

If you're interested in becoming a regular contributor, please reach out to the project maintainers by creating an issue or sending an email to <kapil@example.com>.
EOF

# Create CODE_OF_CONDUCT.md
cat > CODE_OF_CONDUCT.md << 'EOF'
# Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at <kapil@example.com>. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 1.4, available at https://www.contributor-covenant.org/version/1/4/code-of-conduct.html
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
*.out

# Config
config.json
.env
.env.local
.env.dev
.env.prod

# Testing
.pytest_cache/
.coverage
htmlcov/
.hypothesis/

# Distribution
*.tar.gz
*.zip
*.whl

# Local config directory
.chatgpt-product-recommender/

# Frontend build files if they exist
frontend/.next/
frontend/out/
EOF

# Create GitHub workflow
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with flake8
      run: |
        flake8 chatgpt_product_recommender.py commands/ ai_providers/ config/ utils/
    
    - name: Test with pytest
      run: |
        python -m pytest tests/ -v
EOF

# Now add all files to git and commit
git add .
git commit -m "Initial commit: ChatGPT Product Recommender CLI with AI integration"

# Create and push the CLI tool repository
echo "🌐 Creating GitHub repository: $USERNAME/$CLI_REPO"
gh repo create "$USERNAME/$CLI_REPO" --public --description "A CLI tool for generating product recommendations using AI models"

# Add the remote origin
git remote add origin "https://github.com/$USERNAME/$CLI_REPO.git"

# Push to GitHub
echo "⬆️  Pushing CLI tool repository to GitHub..."
git branch -M main
git push -u origin main
echo "✅ CLI tool repository pushed successfully!"

# Create and push the main repository
echo "📦 Creating main repository: $MAIN_REPO"
cd "$PROJECT_ROOT"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    git config user.name "Kapil"
    git config user.email "kapil@example.com"
fi

# Create the main repository on GitHub
echo "🌐 Creating GitHub repository: $USERNAME/$MAIN_REPO"
gh repo create "$USERNAME/$MAIN_REPO" --public --description "Complete product recommendation system with web and CLI interfaces"

# Add the remote origin
git remote add origin "https://github.com/$USERNAME/$MAIN_REPO.git"

# Add all files and commit
git add .
# Remove the cli-tool directory from the main project since we just recreated it separately
if [ -d "cli-tool" ]; then
    rm -rf cli-tool
fi
git add .
git commit -m "Initial commit: Magento Product Recommender with CLI tool reference" || true

# Push to GitHub
echo "⬆️  Pushing main repository to GitHub..."
git branch -M main
git push -u origin main
echo "✅ Main repository pushed successfully!"

echo ""
echo "🎉 All repositories created and pushed successfully!"
echo ""
echo "Repositories:"
echo "  - CLI Tool: https://github.com/$USERNAME/$CLI_REPO"
echo "  - Main Project: https://github.com/$USERNAME/$MAIN_REPO"
echo ""
echo "💡 Next steps:"
echo "  - If you want to link the CLI tool as a submodule in the main repo, run:"
echo "      cd $PROJECT_ROOT"
echo "      git submodule add https://github.com/$USERNAME/$CLI_REPO.git cli-tool"
echo "      git add .gitmodules cli-tool"
echo "      git commit -m \"Add cli-tool as submodule\""
echo "      git push"