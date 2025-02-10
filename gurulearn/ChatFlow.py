import json
import os
import pandas as pd


class FlowBot:


    def __init__(self, data):
        self.df = data.copy()
        self.df_display = data.copy()
        self.df_clean = data.copy()
        for col in self.df_clean.select_dtypes(include='object'):
            self.df_clean[col] = self.df_clean[col].astype(str).str.strip().str.lower()
        self.flow = []
        self.prompts = {}
        self.result_columns = []
        self.sessions = {}
        self.personal_info_fields = {}
        self.chat_history = {}

    def add_personal_info(self, field, prompt, required=True):
        """Add a personal information field to collect from the user"""
        self.personal_info_fields[field] = {
            'prompt': prompt,
            'required': required
        }

    def add(self, field, prompt, required=True):
        """Add a step to the booking flow"""
        if field not in self.df.columns:
            raise ValueError(f"Column '{field}' not found in dataset")
        self.flow.append({
            'field': field,
            'required': required
        })
        self.prompts[field] = prompt

    def finish(self, *result_columns):
        """Set result columns to display in final output"""
        if not result_columns:
            raise ValueError("At least one result column must be specified")
            
        for column in result_columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
                
        self.result_columns = list(result_columns)

    def get_suggestions(self, user_id):
        """Get available options based on current state"""
        session = self.sessions[user_id]
        current_step = session['step']
        filtered = self.df_clean.copy()
        for step in self.flow[:current_step]:
            field = step['field']
            if field in session['selections']:
                val = session['selections'][field]
                if val:
                    filtered = filtered[filtered[field] == val.lower()]
        current_field = self.flow[current_step]['field']
        options = filtered[current_field].unique().tolist()
        display_options = []
        for opt in options:
            if pd.notna(opt):
                mask = self.df_clean[current_field] == opt
                display_val = self.df_display.loc[mask, current_field].iloc[0]
                display_options.append(display_val)
        return [str(opt) for opt in display_options if opt and pd.notna(opt)]

    def _log_interaction(self, user_id, user_input, bot_response):
        """Helper method to log interactions to chat history"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        if bot_response is not None or user_input:
            self.chat_history[user_id].append({
                'user_input': user_input,
                'bot_response': bot_response
            })

    def process(self, user_id, text):
        """Process user input and return response"""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'step': 0,
                'selections': {},
                'completed': False,
                'personal_info': {}
            }
            self.chat_history[user_id] = []
            
        session = self.sessions[user_id]
        if session['completed']:
            self.reset_session(user_id)

        if len(session['personal_info']) < len(self.personal_info_fields):
            response = self._collect_personal_info(user_id, text)
            return response

        current_step = session['step']
        if current_step >= len(self.flow):
            return self._finalize_response(user_id)

        current_field = self.flow[current_step]['field']
        required = self.flow[current_step]['required']

        if not text.strip():
            if required:
                message = f"This field is required. Please choose from: {', '.join(self.get_suggestions(user_id))}"
                self._log_interaction(user_id, "", message)
                response = {
                    'message': message,
                    'suggestions': self.get_suggestions(user_id)
                }
                return response
            else:
                session['selections'][current_field] = None
                session['step'] += 1

        else:
            cleaned_input = str(text).strip().lower()
            available = [str(x).lower() for x in self.get_suggestions(user_id)]
            
            if cleaned_input not in available and text not in self.get_suggestions(user_id):
                if required:
                    message = f"Invalid option. Please choose from: {', '.join(self.get_suggestions(user_id))}"
                    self._log_interaction(user_id, text, message)
                    response = {
                        'message': message,
                        'suggestions': self.get_suggestions(user_id)
                    }
                    return response
                else:
                    session['selections'][current_field] = None
                    session['step'] += 1
            else:
                mask = self.df_display[current_field].astype(str).str.lower() == cleaned_input
                if any(mask):
                    clean_value = self.df_clean.loc[mask, current_field].iloc[0]
                else:
                    clean_value = cleaned_input
                session['selections'][current_field] = clean_value
                session['step'] += 1

        if session['step'] >= len(self.flow):
            self._log_interaction(user_id, text, self._generate_final_message(user_id))
            return self._finalize_response(user_id)

        next_field = self.flow[session['step']]['field']
        next_prompt = self.prompts[next_field]
        response = {
            'message': next_prompt,
            'suggestions': self.get_suggestions(user_id)
        }
        
        self._log_interaction(user_id, text, next_prompt)
        return response

    def _collect_personal_info(self, user_id, text):
        """Collect personal information from the user"""
        session = self.sessions[user_id]
        personal_info = session['personal_info']
        fields = list(self.personal_info_fields.keys())
        
        for i, field in enumerate(fields):
            if field not in personal_info:
                info = self.personal_info_fields[field]
                if not text.strip():
                    response = {
                        'message': info['prompt'],
                        'suggestions': []
                    }
                    if i == 0:
                        self._log_interaction(user_id, "", info['prompt'])
                    return response
                else:
                    personal_info[field] = text.strip()
                    
                    if i + 1 < len(fields):
                        next_field = fields[i + 1]
                        next_prompt = self.personal_info_fields[next_field]['prompt']
                    else:
                        next_prompt = self.prompts[self.flow[0]['field']] if self.flow else None
                    
                    self._log_interaction(user_id, text, next_prompt)
                    
                    if i + 1 < len(fields):
                        return {
                            'message': next_prompt,
                            'suggestions': []
                        }
                    else:
                        session['step'] = 0
                        if self.flow:
                            return {
                                'message': next_prompt,
                                'suggestions': self.get_suggestions(user_id)
                            }
                        else:
                            return self._finalize_response(user_id)
        return self.process(user_id, "")

    def _generate_final_message(self, user_id):
        """Generate the final results message"""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        for field, value in session['selections'].items():
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

    def _finalize_response(self, user_id):
        """Generate final results"""
        session = self.sessions[user_id]
        filtered = self.df_clean.copy()
        for field, value in session['selections'].items():
            if value:
                filtered = filtered[filtered[field] == value]
        results = self.df_display.loc[filtered.index]
        
        final_message = self._generate_final_message(user_id)
        response = {
            'completed': True,
            'results': results[self.result_columns].to_dict('records'),
            'message': final_message
        }
        
        session['completed'] = True
        self._save_to_json(user_id)
        return response

    def _save_to_json(self, user_id):
        """Save chat history and personal info to a JSON file"""
        session = self.sessions[user_id]
        data_to_save = {
            'personal_info': session['personal_info'],
            'chat_history': self.chat_history[user_id]
        }
        if not os.path.exists('user_data'):
            os.makedirs('user_data')
        with open(f'user_data/{user_id}.json', 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def reset_session(self, user_id):
        """Reset user's session"""
        self.sessions[user_id] = {
            'step': 0,
            'selections': {},
            'completed': False,
            'personal_info': {}
        }
        self.chat_history[user_id] = []
