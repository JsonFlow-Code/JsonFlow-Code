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
from typing import Dict, Type, Any
from engine.generator import LanguageGenerator, PluginInterface
from engine.executor import Executor, ExecutorPluginInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Registry:
    """Central registry for generators, executors, and plugins."""
    def __init__(self):
        self.generators: Dict[str, Type[LanguageGenerator]] = {}
        self.executors: Dict[str, Type[Executor]] = {}
        self.plugins: Dict[str, PluginInterface] = {}
        self.executor_plugins: Dict[str, ExecutorPluginInterface] = {}

    def register_generator(self, language: str, generator_class: Type[LanguageGenerator]) -> None:
        """
        Register a language-specific generator.

        Args:
            language: Language identifier (e.g., 'python').
            generator_class: Generator class implementing LanguageGenerator.
        """
        if language in self.generators:
            logger.warning(f"Overriding generator for language: {language}")
        self.generators[language] = generator_class
        logger.info(f"Registered generator for language: {language}")

    def register_executor(self, language: str, executor_class: Type[Executor]) -> None:
        """
        Register a language-specific executor.

        Args:
            language: Language identifier (e.g., 'python').
            executor_class: Executor class implementing Executor.
        """
        if language in self.executors:
            logger.warning(f"Overriding executor for language: {language}")
        self.executors[language] = executor_class
        logger.info(f"Registered executor for language: {language}")

    def register_plugin(self, plugin_name: str, plugin_class: Type[PluginInterface]) -> None:
        """
        Register a generator plugin.

        Args:
            plugin_name: Unique name for the plugin.
            plugin_class: Plugin class implementing PluginInterface.
        """
        if plugin_name in self.plugins:
            logger.warning(f"Overriding plugin: {plugin_name}")
        self.plugins[plugin_name] = plugin_class()
        logger.info(f"Registered plugin: {plugin_name}")

    def register_executor_plugin(self, plugin_name: str, plugin_class: Type[ExecutorPluginInterface]) -> None:
        """
        Register an executor plugin.

        Args:
            plugin_name: Unique name for the plugin.
            plugin_class: Plugin class implementing ExecutorPluginInterface.
        """
        if plugin_name in self.executor_plugins:
            logger.warning(f"Overriding executor plugin: {plugin_name}")
        self.executor_plugins[plugin_name] = plugin_class()
        logger.info(f"Registered executor plugin: {plugin_name}")

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

registry = Registry()
