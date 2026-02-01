import os
import re
from typing import Optional
from enum import Enum

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message


ALLOWED_LABELS = {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}
DEFAULT_LABEL = "Error - no LLM configured"
DEFAULT_TIMEOUT = 30.0


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    NONE = "none"


class Agent:
    """
    Purple Agent for RIT classification.
    
    Supported providers (checked in order):
    - Anthropic: ANTHROPIC_API_KEY
    - Google Gemini: GOOGLE_API_KEY
    - xAI Grok: XAI_API_KEY
    - OpenAI/OpenRouter: OPENAI_API_KEY or LLM_API_KEY
    
    Without API key: returns default label for all requests (test mode)
    """
    
    def __init__(self):
        self.provider = Provider.NONE
        self.client = None
        self.model = None
        self.timeout = float(os.environ.get("LLM_TIMEOUT", DEFAULT_TIMEOUT))
        
        self._detect_and_setup_provider()

    def _detect_and_setup_provider(self):
        """Detect available API key and setup the appropriate client."""
        
        # Check Anthropic first
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._setup_anthropic(anthropic_key)
            return
        
        # Check Google Gemini
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            self._setup_google(google_key)
            return
        
        # Check xAI Grok
        xai_key = os.environ.get("XAI_API_KEY")
        if xai_key:
            self._setup_xai(xai_key)
            return
        
        # Check OpenAI / OpenRouter / generic LLM_API_KEY
        openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if openai_key:
            self._setup_openai(openai_key)
            return
        
        print(f"INFO: No API key found. Test mode: returning '{DEFAULT_LABEL}' for all requests.")

    def _setup_openai(self, api_key: str):
        """Initialize OpenAI-compatible client (works for OpenRouter too)."""
        try:
            from openai import AsyncOpenAI
            
            api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE")
            self.model = os.environ.get("LLM_MODEL", "gpt-4o")
            
            client_kwargs = {"api_key": api_key, "timeout": self.timeout}
            if api_base:
                client_kwargs["base_url"] = api_base
            
            self.client = AsyncOpenAI(**client_kwargs)
            self.provider = Provider.OPENAI
            print(f"INFO: OpenAI provider configured - model: {self.model}, base: {api_base or 'default'}")
            
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")

    def _setup_anthropic(self, api_key: str):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            
            self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
            self.client = AsyncAnthropic(api_key=api_key, timeout=self.timeout)
            self.provider = Provider.ANTHROPIC
            print(f"INFO: Anthropic provider configured - model: {self.model}")
            
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")

    def _setup_google(self, api_key: str):
        """Initialize Google Gemini client."""
        try:
            from google import genai
            
            self.model = os.environ.get("LLM_MODEL", "gemini-2.5-pro")
            self.client = genai.Client(api_key=api_key)
            self.provider = Provider.GOOGLE
            print(f"INFO: Google Gemini provider configured - model: {self.model}")
            
        except ImportError:
            print("ERROR: google-genai package not installed. Run: pip install google-genai")

    def _setup_xai(self, api_key: str):
        """Initialize xAI Grok client (OpenAI-compatible endpoint)."""
        try:
            from openai import AsyncOpenAI
            
            self.model = os.environ.get("LLM_MODEL", "grok-4")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                timeout=self.timeout,
            )
            self.provider = Provider.XAI
            print(f"INFO: xAI Grok provider configured - model: {self.model}")
            
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")

    def _parse_payload(self, text: str) -> tuple[str, str]:
        """Parse Green Agent payload format."""
        rules_match = re.search(
            r"={3,}\s*RULES\s*START\s*={3,}\s*\n(.*?)\n\s*={3,}\s*RULES\s*END\s*={3,}",
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if rules_match:
            ruleset = rules_match.group(1).strip()
            prompt_end = text.find("===== RULES START =====")
            prompt = text[:prompt_end].strip() if prompt_end != -1 else ""
        else:
            prompt = ""
            ruleset = text.strip()
        
        return prompt, ruleset

    def _extract_label(self, response: str) -> Optional[str]:
        """Extract valid label from LLM response."""
        text = (response or "").upper()
        hits = [lab for lab in ALLOWED_LABELS if re.search(rf"\b{lab}\b", text)]
        if len(hits) == 1:
            return hits[0]
        if text.strip() in ALLOWED_LABELS:
            return text.strip()
        return None

    def _build_messages(self, prompt: str, ruleset: str) -> tuple[str, str]:
        """Build system and user messages for classification."""
        system_msg = (
            "You are an expert at classifying openHAB rules for Rule Interaction Threats (RIT). "
            "Respond with exactly one label: WAC, SAC, WTC, STC, WCC, or SCC. "
            "Output only the label, nothing else."
        )
        user_msg = f"{prompt}\n\nRuleset:\n{ruleset}" if prompt else f"Classify this ruleset:\n{ruleset}"
        return system_msg, user_msg

    async def _classify_openai(self, prompt: str, ruleset: str) -> str:
        """Classify using OpenAI-compatible API."""
        system_msg, user_msg = self._build_messages(prompt, ruleset)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=50,
            temperature=0,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    async def _classify_anthropic(self, prompt: str, ruleset: str) -> str:
        """Classify using Anthropic API."""
        system_msg, user_msg = self._build_messages(prompt, ruleset)
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=50,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
            timeout=self.timeout,
        )
        return response.content[0].text if response.content else ""

    async def _classify_google(self, prompt: str, ruleset: str) -> str:
        """Classify using Google Gemini API."""
        import asyncio
        system_msg, user_msg = self._build_messages(prompt, ruleset)
        
        full_prompt = f"{system_msg}\n\n{user_msg}"
        
        # google-genai uses sync API, wrap in executor with timeout
        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=full_prompt,
                    )
                ),
                timeout=self.timeout,
            )
            return response.text if response.text else ""
        except asyncio.TimeoutError:
            raise TimeoutError(f"Google Gemini request timed out after {self.timeout}s")

    async def _classify_xai(self, prompt: str, ruleset: str) -> str:
        """Classify using xAI Grok API (OpenAI-compatible)."""
        system_msg, user_msg = self._build_messages(prompt, ruleset)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=50,
            temperature=0,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    async def _classify(self, prompt: str, ruleset: str) -> str:
        """Route to appropriate provider and handle response."""
        if self.provider == Provider.NONE or not self.client:
            return DEFAULT_LABEL
        
        try:
            if self.provider == Provider.OPENAI:
                result = await self._classify_openai(prompt, ruleset)
            elif self.provider == Provider.ANTHROPIC:
                result = await self._classify_anthropic(prompt, ruleset)
            elif self.provider == Provider.GOOGLE:
                result = await self._classify_google(prompt, ruleset)
            elif self.provider == Provider.XAI:
                result = await self._classify_xai(prompt, ruleset)
            else:
                return DEFAULT_LABEL
            
            label = self._extract_label(result)
            if label:
                return label
            
            # Could not parse valid label - log and return default
            print(f"WARN: Could not parse label from LLM response: {result[:100]}")
            return DEFAULT_LABEL
            
        except Exception as e:
            print(f"ERROR: LLM call failed ({self.provider.value}): {e}")
            return DEFAULT_LABEL

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Handle incoming classification requests."""
        input_text = get_message_text(message)
        
        # Health check
        if "health check" in input_text.lower():
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="OK"))],
                name="HealthCheck",
            )
            return
        
        # No provider configured = test mode
        if self.provider == Provider.NONE or not self.client:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=DEFAULT_LABEL))],
                name="Classification",
            )
            return
        
        # Classification request
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Classifying with {self.provider.value}/{self.model}...")
        )
        
        prompt, ruleset = self._parse_payload(input_text)
        result = await self._classify(prompt, ruleset)
        
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="Classification",
        )