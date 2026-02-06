"""
ChatFlow - Conversational flow management for chatbots.

Provides a simple way to create guided conversations with branching logic,
data filtering, and session management.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


class FlowResponse(TypedDict, total=False):
    """Response structure from flow processing."""
    message: str
    suggestions: list[str]
    completed: bool
    results: list[dict[str, Any]]


@dataclass
class SessionState:
    """State for a user session."""
    step: int = 0
    selections: dict[str, str | None] = field(default_factory=dict)
    completed: bool = False
    personal_info: dict[str, str] = field(default_factory=dict)


class FlowBot:
    """
    A conversational flow bot for guided data exploration and booking.
    
    Args:
        data: DataFrame containing the data to filter through
        data_dir: Directory for saving user session data (default: 'user_data')
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'A', 'B'],
        ...     'name': ['Item1', 'Item2', 'Item3'],
        ...     'price': [10, 20, 15]
        ... })
        >>> bot = FlowBot(df)
        >>> bot.add('category', 'Select a category:')
        >>> bot.finish('name', 'price')
        >>> response = bot.process('user123', '')
    """

    def __init__(self, data: pd.DataFrame, data_dir: str | Path = "user_data"):
        self.df = data.copy()
        self.df_display = data.copy()
        self.df_clean = data.copy()
        self.data_dir = Path(data_dir)
        
        # Normalize string columns for matching
        for col in self.df_clean.select_dtypes(include="object"):
            self.df_clean[col] = self.df_clean[col].astype(str).str.strip().str.lower()
        
        self.flow: list[dict[str, Any]] = []
        self.prompts: dict[str, str] = {}
        self.result_columns: list[str] = []
        self.sessions: dict[str, SessionState] = {}
        self.personal_info_fields: dict[str, dict[str, Any]] = {}
        self.chat_history: dict[str, list[dict[str, Any]]] = {}

    def add_personal_info(self, field_name: str, prompt: str, required: bool = True) -> "FlowBot":
        """
        Add a personal information field to collect from the user.
        
        Args:
            field_name: Name of the field to collect
            prompt: Question to ask the user
            required: Whether the field is required
            
        Returns:
            Self for method chaining
        """
        self.personal_info_fields[field_name] = {
            "prompt": prompt,
            "required": required
        }
        return self

    def add(self, field: str, prompt: str, required: bool = True) -> "FlowBot":
        """
        Add a step to the conversation flow.
        
        Args:
            field: Column name in the DataFrame to filter on
            prompt: Question to ask the user
            required: Whether a selection is required
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If field is not in the DataFrame
        """
        if field not in self.df.columns:
            raise ValueError(f"Column '{field}' not found in dataset. Available: {list(self.df.columns)}")
        
        self.flow.append({
            "field": field,
            "required": required
        })
        self.prompts[field] = prompt
        return self

    def finish(self, *result_columns: str) -> "FlowBot":
        """
        Set which columns to display in the final results.
        
        Args:
            *result_columns: Column names to include in results
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If no columns specified or column not found
        """
        if not result_columns:
            raise ValueError("At least one result column must be specified")
        
        for column in result_columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset. Available: {list(self.df.columns)}")
        
        self.result_columns = list(result_columns)
        return self

    def validate(self) -> list[str]:
        """
        Validate the flow configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.flow:
            errors.append("No flow steps defined. Use add() to add steps.")
        
        if not self.result_columns:
            errors.append("No result columns defined. Use finish() to set result columns.")
        
        return errors

    def get_suggestions(self, user_id: str) -> list[str]:
        """
        Get available options based on current session state.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of available option strings
        """
        session = self.sessions.get(user_id)
        if not session:
            return []
        
        current_step = session.step
        if current_step >= len(self.flow):
            return []
        
        # Filter data based on previous selections
        filtered = self.df_clean.copy()
        for step in self.flow[:current_step]:
            field = step["field"]
            val = session.selections.get(field)
            if val:
                filtered = filtered[filtered[field] == val.lower()]
        
        current_field = self.flow[current_step]["field"]
        options = filtered[current_field].unique().tolist()
        
        # Get display values
        display_options = []
        for opt in options:
            if pd.notna(opt):
                mask = self.df_clean[current_field] == opt
                if mask.any():
                    display_val = self.df_display.loc[mask, current_field].iloc[0]
                    display_options.append(str(display_val))
        
        return [opt for opt in display_options if opt and pd.notna(opt)]

    def _log_interaction(self, user_id: str, user_input: str, bot_response: str | None) -> None:
        """Log an interaction to chat history."""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        if bot_response is not None or user_input:
            self.chat_history[user_id].append({
                "user_input": user_input,
                "bot_response": bot_response
            })

    def process(self, user_id: str, text: str) -> FlowResponse:
        """
        Process user input and return the next response.
        
        Args:
            user_id: User identifier
            text: User's input text
            
        Returns:
            FlowResponse with message, suggestions, and optionally results
        """
        # Initialize session if needed
        if user_id not in self.sessions:
            self.sessions[user_id] = SessionState()
            self.chat_history[user_id] = []
        
        session = self.sessions[user_id]
        
        # Reset if completed
        if session.completed:
            self.reset_session(user_id)
            session = self.sessions[user_id]

        # Collect personal info first
        if len(session.personal_info) < len(self.personal_info_fields):
            return self._collect_personal_info(user_id, text)

        current_step = session.step
        if current_step >= len(self.flow):
            return self._finalize_response(user_id)

        current_field = self.flow[current_step]["field"]
        required = self.flow[current_step]["required"]

        # Handle empty input
        if not text.strip():
            if required:
                suggestions = self.get_suggestions(user_id)
                message = f"This field is required. Please choose from: {', '.join(suggestions)}"
                self._log_interaction(user_id, "", message)
                return FlowResponse(message=message, suggestions=suggestions)
            else:
                session.selections[current_field] = None
                session.step += 1
        else:
            # Validate input
            cleaned_input = str(text).strip().lower()
            available = [str(x).lower() for x in self.get_suggestions(user_id)]
            
            if cleaned_input not in available and text not in self.get_suggestions(user_id):
                if required:
                    suggestions = self.get_suggestions(user_id)
                    message = f"Invalid option. Please choose from: {', '.join(suggestions)}"
                    self._log_interaction(user_id, text, message)
                    return FlowResponse(message=message, suggestions=suggestions)
                else:
                    session.selections[current_field] = None
                    session.step += 1
            else:
                # Valid input - store and advance
                mask = self.df_display[current_field].astype(str).str.lower() == cleaned_input
                if mask.any():
                    clean_value = self.df_clean.loc[mask, current_field].iloc[0]
                else:
                    clean_value = cleaned_input
                session.selections[current_field] = clean_value
                session.step += 1

        # Check if flow is complete
        if session.step >= len(self.flow):
            self._log_interaction(user_id, text, self._generate_final_message(user_id))
            return self._finalize_response(user_id)

        # Return next prompt
        next_field = self.flow[session.step]["field"]
        next_prompt = self.prompts[next_field]
        suggestions = self.get_suggestions(user_id)
        
        self._log_interaction(user_id, text, next_prompt)
        return FlowResponse(message=next_prompt, suggestions=suggestions)

    def _collect_personal_info(self, user_id: str, text: str) -> FlowResponse:
        """Collect personal information from the user."""
        session = self.sessions[user_id]
        personal_info = session.personal_info
        fields = list(self.personal_info_fields.keys())
        
        for i, field_name in enumerate(fields):
            if field_name not in personal_info:
                info = self.personal_info_fields[field_name]
                
                if not text.strip():
                    if i == 0:
                        self._log_interaction(user_id, "", info["prompt"])
                    return FlowResponse(message=info["prompt"], suggestions=[])
                else:
                    personal_info[field_name] = text.strip()
                    
                    if i + 1 < len(fields):
                        next_field = fields[i + 1]
                        next_prompt = self.personal_info_fields[next_field]["prompt"]
                        self._log_interaction(user_id, text, next_prompt)
                        return FlowResponse(message=next_prompt, suggestions=[])
                    else:
                        # All personal info collected, start flow
                        session.step = 0
                        if self.flow:
                            first_prompt = self.prompts[self.flow[0]["field"]]
                            self._log_interaction(user_id, text, first_prompt)
                            return FlowResponse(
                                message=first_prompt,
                                suggestions=self.get_suggestions(user_id)
                            )
                        else:
                            return self._finalize_response(user_id)
        
        return self.process(user_id, "")

    def _generate_final_message(self, user_id: str) -> str:
        """Generate the final results message."""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        
        for field, value in session.selections.items():
            if value:
                filtered = filtered[filtered[field] == value]
        
        results = self.df_display.loc[filtered.index]
        
        if len(results) == 0:
            return "No results found matching your criteria"
        
        final_message = f"Found {len(results)} matching options:\n"
        for _, row in results.iterrows():
            result_items = [f"{col}: {row[col]}" for col in self.result_columns]
            final_message += f"- {' | '.join(result_items)}\n"
        
        return final_message

    def _finalize_response(self, user_id: str) -> FlowResponse:
        """Generate final results and mark session as complete."""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        
        for field, value in session.selections.items():
            if value:
                filtered = filtered[filtered[field] == value]
        
        results = self.df_display.loc[filtered.index]
        final_message = self._generate_final_message(user_id)
        
        response = FlowResponse(
            completed=True,
            results=results[self.result_columns].to_dict("records"),
            message=final_message
        )
        
        session.completed = True
        self._save_to_json(user_id)
        return response

    def _save_to_json(self, user_id: str) -> None:
        """Save chat history and personal info to a JSON file."""
        session = self.sessions[user_id]
        data_to_save = {
            "personal_info": session.personal_info,
            "selections": session.selections,
            "chat_history": self.chat_history.get(user_id, [])
        }
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = self.data_dir / f"{user_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def reset_session(self, user_id: str) -> None:
        """Reset a user's session to start fresh."""
        self.sessions[user_id] = SessionState()
        self.chat_history[user_id] = []

    def get_session_data(self, user_id: str) -> dict[str, Any] | None:
        """
        Get current session data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session data dict or None if no session exists
        """
        if user_id not in self.sessions:
            return None
        
        session = self.sessions[user_id]
        return {
            "step": session.step,
            "selections": session.selections,
            "completed": session.completed,
            "personal_info": session.personal_info,
            "chat_history": self.chat_history.get(user_id, [])
        }

    def export_history(self, user_id: str, format: str = "json") -> str | pd.DataFrame:
        """
        Export chat history in various formats.
        
        Args:
            user_id: User identifier
            format: Output format ('json' or 'dataframe')
            
        Returns:
            JSON string or DataFrame depending on format
        """
        history = self.chat_history.get(user_id, [])
        
        if format == "dataframe":
            return pd.DataFrame(history)
        else:
            return json.dumps(history, indent=2, ensure_ascii=False)
