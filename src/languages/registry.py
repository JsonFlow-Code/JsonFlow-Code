# Copyright (c) 2024 James Chapman
#
# This software is dual-licensed:
#
# - For individuals and non-commercial use: Licensed under the MIT License.
# - For commercial or corporate use: A separate commercial license is required.
#
# To obtain a commercial license, please contact: iconoclastdao@gmail.com
#
# By using this software, you agree to these terms.
#
# MIT License (for individuals and non-commercial use):
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import importlib
import pkgutil
from typing import Dict, Type, Any, Optional
from engine.generator import LanguageGenerator, PluginInterface
from engine.executor import Executor, ExecutorPluginInterface
from engine.workflow import Workflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LanguageParserInterface:
    """Interface for language-specific parsers (code to Workflow)."""
    def parse(self, code: str, metadata: Dict[str, Any]) -> Workflow:
        """
        Parse language-specific code into a Workflow object.

        Args:
            code: The code to parse.
            metadata: Metadata for the workflow (e.g., target_language).

        Returns:
            Workflow: Parsed Workflow object.

        Raises:
            ValueError: If parsing fails.
        """
        raise NotImplementedError("parse method must be implemented")

class LanguageValidatorInterface:
    """Interface for language-specific validators."""
    def validate(self, code: str) -> None:
        """
        Validate language-specific code.

        Args:
            code: The code to validate.

        Raises:
            ValueError: If validation fails.
        """
        raise NotImplementedError("validate method must be implemented")

class Registry:
    """Central registry for language and plugin components."""
    def __init__(self):
        self.generators: Dict[str, Type[LanguageGenerator]] = {}
        self.executors: Dict[str, Type[Executor]] = {}
        self.parsers: Dict[str, Type[LanguageParserInterface]] = {}
        self.validators: Dict[str, Type[LanguageValidatorInterface]] = {}
        self.plugins: Dict[str, PluginInterface] = {}
        self.executor_plugins: Dict[str, ExecutorPluginInterface] = {}

    def register_language(self, language: str, module: Any) -> None:
        """
        Register a language module with its components.

        Args:
            language: Language identifier (e.g., 'python').
            module: Module containing Generator, Executor, Parser, and Validator classes.
        """
        try:
            if hasattr(module, 'Generator'):
                self.generators[language] = module.Generator
                logger.info(f"Registered generator for language: {language}")
            if hasattr(module, 'Executor'):
                self.executors[language] = module.Executor
                logger.info(f"Registered executor for language: {language}")
            if hasattr(module, 'Parser'):
                self.parsers[language] = module.Parser
                logger.info(f"Registered parser for language: {language}")
            if hasattr(module, 'Validator'):
                self.validators[language] = module.Validator
                logger.info(f"Registered validator for language: {language}")
        except Exception as e:
            logger.error(f"Failed to register language {language}: {str(e)}", exc_info=True)

    def register_plugin(self, plugin_name: str, plugin_class: Type[PluginInterface], executor_plugin_class: Optional[Type[ExecutorPluginInterface]] = None) -> None:
        """
        Register a plugin and its executor plugin.

        Args:
            plugin_name: Unique name for the plugin.
            plugin_class: Plugin class implementing PluginInterface.
            executor_plugin_class: Executor plugin class implementing ExecutorPluginInterface, if any.
        """
        try:
            self.plugins[plugin_name] = plugin_class()
            logger.info(f"Registered plugin: {plugin_name}")
            if executor_plugin_class:
                self.executor_plugins[plugin_name] = executor_plugin_class()
                logger.info(f"Registered executor plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_name}: {str(e)}", exc_info=True)

    def load_languages(self, language_dir: str = "languages") -> None:
        """
        Dynamically load all language modules from the languages directory.

        Args:
            language_dir: Directory containing language modules.
        """
        try:
            package = importlib.import_module(language_dir)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                if module_name == "registry" or module_name.startswith("plugins"):
                    continue
                try:
                    module = importlib.import_module(f"{language_dir}.{module_name}")
                    language = module_name
                    self.register_language(language, module)
                except Exception as e:
                    logger.error(f"Failed to load language module {module_name}: {str(e)}", exc_info=True)
        except ImportError as e:
            logger.error(f"Failed to import language package {language_dir}: {str(e)}", exc_info=True)

    def load_plugins(self, plugin_dir: str = "languages.plugins") -> None:
        """
        Dynamically load all plugin modules from the plugins directory.

        Args:
            plugin_dir: Directory containing plugin modules.
        """
        try:
            package = importlib.import_module(plugin_dir)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{plugin_dir}.{module_name}")
                    plugin_name = module_name
                    plugin_class = getattr(module, "Plugin", None)
                    executor_plugin_class = getattr(module, "ExecutorPlugin", None)
                    if plugin_class:
                        self.register_plugin(plugin_name, plugin_class, executor_plugin_class)
                    else:
                        logger.warning(f"No Plugin class found in {module_name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin module {module_name}: {str(e)}", exc_info=True)
        except ImportError as e:
            logger.error(f"Failed to import plugin package {plugin_dir}: {str(e)}", exc_info=True)

    def get_generator(self, language: str) -> LanguageGenerator:
        """
        Retrieve a generator for the specified language.

        Args:
            language: Language identifier.

        Returns:
            LanguageGenerator: Instantiated generator.

        Raises:
            ValueError: If language is not supported.
        """
        if language not in self.generators:
            logger.error(f"Unsupported language: {language}")
            raise ValueError(f"Unsupported language: {language}")
        return self.generators[language]()

    def get_executor(self, language: str) -> Executor:
        """
        Retrieve an executor for the specified language.

        Args:
            language: Language identifier.

        Returns:
            Executor: Instantiated executor.

        Raises:
            ValueError: If language is not supported.
        """
        if language not in self.executors:
            logger.error(f"Unsupported language: {language}")
            raise ValueError(f"Unsupported language: {language}")
        return self.executors[language]()

    def get_parser(self, language: str) -> LanguageParserInterface:
        """
        Retrieve a parser for the specified language.

        Args:
            language: Language identifier.

        Returns:
            LanguageParserInterface: Instantiated parser.

        Raises:
            ValueError: If language is not supported.
        """
        if language not in self.parsers:
            logger.error(f"Unsupported parser for language: {language}")
            raise ValueError(f"Unsupported parser for language: {language}")
        return self.parsers[language]()

    def get_validator(self, language: str) -> Optional[LanguageValidatorInterface]:
        """
        Retrieve a validator for the specified language, if available.

        Args:
            language: Language identifier.

        Returns:
            Optional[LanguageValidatorInterface]: Instantiated validator or None.
        """
        return self.validators.get(language, lambda: None)()

# Singleton registry instance
registry = Registry()

# Load all languages and plugins on import
registry.load_languages()
registry.load_plugins()
