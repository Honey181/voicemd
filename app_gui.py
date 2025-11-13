"""
VoiceMD - Standalone Application
Modern interface for voice analysis
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from typing import Optional
from app_predictor import VoiceAnalyzer


# Modern color scheme
COLORS = {
    'bg': '#F3F3F3',
    'card': '#FCFCFC',
    'accent': '#0078D4',
    'success': '#107C10',
    'warning': '#F7630C',
    'error': '#D13438',
    'text': '#000000',
    'text_secondary': '#605E5C',
    'border': '#E5E5E5',
    'divider': '#EDEBE9',
    'input_bg': '#FFFFFF'
}

FONTS = {
    'title': ('Segoe UI', 20, 'bold'),
    'subtitle': ('Segoe UI', 9),
    'heading': ('Segoe UI', 11, 'bold'),
    'body': ('Segoe UI', 10),
    'body_bold': ('Segoe UI', 10, 'bold'),
    'small': ('Segoe UI', 9),
    'mono': ('Consolas', 10)
}


class RoundedButton(tk.Canvas):
    """Modern rounded button"""
    def __init__(self, parent, text, command, bg_color, fg_color='white', **kwargs):
        super().__init__(parent, height=36, bg=parent['bg'], 
                        highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.is_enabled = True
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<Configure>', self._draw)
        self._draw()
    
    def _draw(self, event=None):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2: w = 100
        if h < 2: h = 36
        
        r = 4
        color = self.bg_color if self.is_enabled else '#CCCCCC'
        
        self.create_polygon(
            [r, 0, w-r, 0, w, 0, w, r, w, h-r, w, h, w-r, h, 
             r, h, 0, h, 0, h-r, 0, r, 0, 0, r, 0],
            fill=color, smooth=True, outline=''
        )
        
        self.create_text(w/2, h/2, text=self.text, 
                        fill=self.fg_color, 
                        font=FONTS['body_bold'])
    
    def _on_click(self, e):
        if self.is_enabled and self.command:
            self.command()
    
    def disable(self):
        self.is_enabled = False
        self._draw()
    
    def enable(self):
        self.is_enabled = True
        self._draw()


class FlatDropdown(tk.Frame):
    """Modern flat dropdown"""
    def __init__(self, parent, values, command, **kwargs):
        super().__init__(parent, bg=COLORS['input_bg'], 
                        highlightbackground=COLORS['border'],
                        highlightthickness=1, **kwargs)
        
        self.values = values
        self.command = command
        self.current_value = values[0] if values else ""
        self.menu_open = False
        
        # Main container
        self.inner = tk.Frame(self, bg=COLORS['input_bg'])
        self.inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Selected value label
        self.label = tk.Label(self.inner,
                             text=self.current_value,
                             font=FONTS['body'],
                             bg=COLORS['input_bg'],
                             fg=COLORS['text'],
                             anchor='w',
                             padx=12,
                             pady=8)
        self.label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Arrow
        self.arrow = tk.Label(self.inner,
                             text="▼",
                             font=('Segoe UI', 8),
                             bg=COLORS['input_bg'],
                             fg=COLORS['accent'],
                             padx=10)
        self.arrow.pack(side=tk.RIGHT)
        
        # Bind click
        self.bind('<Button-1>', self._show_menu)
        self.label.bind('<Button-1>', self._show_menu)
        self.arrow.bind('<Button-1>', self._show_menu)
        
        # Popup menu
        self.popup = None
    
    def _show_menu(self, event=None):
        """Show dropdown menu"""
        if self.menu_open:
            return
        
        self.menu_open = True
        
        # Create popup
        self.popup = tk.Toplevel(self)
        self.popup.overrideredirect(True)
        self.popup.configure(bg=COLORS['input_bg'],
                            highlightbackground=COLORS['border'],
                            highlightthickness=1)
        
        # Position below dropdown
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        self.popup.geometry(f"+{x}+{y}")
        
        # Menu items
        for value in self.values:
            item = tk.Label(self.popup,
                          text=value,
                          font=FONTS['body'],
                          bg=COLORS['input_bg'],
                          fg=COLORS['text'],
                          anchor='w',
                          padx=12,
                          pady=8)
            item.pack(fill=tk.X)
            
            # Hover effect
            def on_enter(e, item=item):
                item.config(bg=COLORS['accent'], fg='white')
            
            def on_leave(e, item=item):
                item.config(bg=COLORS['input_bg'], fg=COLORS['text'])
            
            def on_click(e, val=value):
                self._select(val)
            
            item.bind('<Enter>', on_enter)
            item.bind('<Leave>', on_leave)
            item.bind('<Button-1>', on_click)
        
        # Close on focus loss
        self.popup.bind('<FocusOut>', lambda e: self._close_menu())
        self.popup.focus_set()
    
    def _select(self, value):
        """Select value"""
        self.current_value = value
        self.label.config(text=value)
        self._close_menu()
        if self.command:
            self.command(value)
    
    def _close_menu(self):
        """Close menu"""
        if self.popup:
            self.popup.destroy()
            self.popup = None
        self.menu_open = False
    
    def get(self):
        """Get current value"""
        return self.current_value
    
    def set(self, value):
        """Set current value"""
        if value in self.values:
            self.current_value = value
            self.label.config(text=value)


class ModernVoiceMDApp:
    """Modern VoiceMD Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("VoiceMD")
        self.root.geometry("950x750")
        self.root.configure(bg=COLORS['bg'])
        self.root.minsize(900, 700)
        
        # State
        self.analyzer: Optional[VoiceAnalyzer] = None
        self.current_file: Optional[str] = None
        self.is_analyzing = False
        self.available_models = []
        self.current_model_info = None
        
        self._create_ui()
        self._load_models()
        self._init_analyzer()
    
    def _create_ui(self):
        """Create compact UI"""
        
        # Main container with less padding
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # ===================
        # COMPACT HEADER
        # ===================
        header = tk.Frame(main, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 15))
        
        # Title and status on same line
        tk.Label(header,
                text="🎤 VoiceMD",
                font=FONTS['title'],
                bg=COLORS['bg'],
                fg=COLORS['text']).pack(side=tk.LEFT)
        
        self.status_badge = tk.Label(header,
                                     text="● Initializing",
                                     font=FONTS['small'],
                                     bg=COLORS['bg'],
                                     fg=COLORS['text_secondary'],
                                     padx=10)
        self.status_badge.pack(side=tk.RIGHT)
        
        # ===================
        # COMPACT CONTROLS CARD
        # ===================
        controls_card = tk.Frame(main, bg=COLORS['card'], 
                                relief=tk.FLAT,
                                highlightbackground=COLORS['border'],
                                highlightthickness=1)
        controls_card.pack(fill=tk.X, pady=(0, 15))
        
        # All controls in one compact card
        controls_inner = tk.Frame(controls_card, bg=COLORS['card'])
        controls_inner.pack(fill=tk.X, padx=15, pady=12)
        
        # ROW 1: Model selection
        row1 = tk.Frame(controls_inner, bg=COLORS['card'])
        row1.pack(fill=tk.X, pady=(0, 4))
        
        tk.Label(row1,
                text="Model:",
                font=FONTS['body_bold'],
                bg=COLORS['card'],
                fg=COLORS['text'],
                width=6,
                anchor='w').pack(side=tk.LEFT, padx=(0, 8))
        
        # Modern flat dropdown
        self.model_dropdown = FlatDropdown(row1,
                                          values=["Loading..."],
                                          command=self._on_model_selected,
                                          width=460,
                                          height=38)
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        
        self.switch_btn = RoundedButton(row1,
                                       text="Switch",
                                       command=self._switch_model,
                                       bg_color=COLORS['accent'],
                                       width=90)
        self.switch_btn.pack(side=tk.LEFT)
        self.switch_btn.disable()
        
        # Model description (aligned with dropdown)
        self.model_desc = tk.Label(controls_inner,
                                   text="",
                                   font=FONTS['small'],
                                   bg=COLORS['card'],
                                   fg=COLORS['text_secondary'],
                                   anchor='w',
                                   justify=tk.LEFT)
        self.model_desc.pack(fill=tk.X, pady=(2, 8), padx=(66, 0))
        
        # ROW 2: File selection
        row2 = tk.Frame(controls_inner, bg=COLORS['card'])
        row2.pack(fill=tk.X)
        
        tk.Label(row2,
                text="File:",
                font=FONTS['body_bold'],
                bg=COLORS['card'],
                fg=COLORS['text'],
                width=6,
                anchor='w').pack(side=tk.LEFT, padx=(0, 8))
        
        self.file_label = tk.Label(row2,
                                   text="No file selected",
                                   font=FONTS['body'],
                                   bg='#F5F5F5',
                                   fg=COLORS['text_secondary'],
                                   anchor='w',
                                   padx=12,
                                   pady=8,
                                   width=40,
                                   relief=tk.FLAT,
                                   borderwidth=1)
        self.file_label.pack(side=tk.LEFT, padx=(0, 8))
        
        # Buttons container for perfect alignment
        btn_frame = tk.Frame(row2, bg=COLORS['card'])
        btn_frame.pack(side=tk.LEFT)
        
        self.select_btn = RoundedButton(btn_frame,
                                       text="Browse",
                                       command=self._select_file,
                                       bg_color=COLORS['accent'],
                                       width=90)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.analyze_btn = RoundedButton(btn_frame,
                                        text="Analyze",
                                        command=self._analyze,
                                        bg_color=COLORS['success'],
                                        width=90)
        self.analyze_btn.pack(side=tk.LEFT)
        self.analyze_btn.disable()
        
        # Progress (hidden by default)
        self.progress_label = tk.Label(controls_card,
                                      text="",
                                      font=FONTS['small'],
                                      bg=COLORS['card'],
                                      fg=COLORS['warning'])
        
        # ===================
        # RESULTS CARD (MAXIMUM SPACE)
        # ===================
        results_card = tk.Frame(main, bg=COLORS['card'],
                               relief=tk.FLAT,
                               highlightbackground=COLORS['border'],
                               highlightthickness=1)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        # Minimal header
        results_header = tk.Frame(results_card, bg=COLORS['card'])
        results_header.pack(fill=tk.X, padx=15, pady=(10, 8))
        
        tk.Label(results_header,
                text="Results",
                font=FONTS['heading'],
                bg=COLORS['card'],
                fg=COLORS['text']).pack(side=tk.LEFT)
        
        # Divider
        tk.Frame(results_card, height=1, bg=COLORS['divider']).pack(fill=tk.X, padx=15)
        
        # Results text (ALL REMAINING SPACE)
        text_frame = tk.Frame(results_card, bg=COLORS['card'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(text_frame,
                                   font=FONTS['mono'],
                                   bg='#FAFAFA',
                                   fg=COLORS['text'],
                                   relief=tk.FLAT,
                                   padx=15,
                                   pady=15,
                                   wrap=tk.WORD,
                                   spacing1=2,
                                   spacing2=1,
                                   spacing3=2,
                                   yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Welcome message
        self.results_text.insert('1.0',
            "Welcome to VoiceMD\n\n"
            "Get started:\n"
            "  1. Select a model (optional)\n"
            "  2. Browse for an audio file\n"
            "  3. Click Analyze\n\n"
            "Supported: WAV, MP3, OGG, FLAC, M4A\n"
            "All analysis happens offline on your device.")
        self.results_text.config(state=tk.DISABLED)
    
    def _load_models(self):
        """Load models"""
        self.available_models = VoiceAnalyzer.get_available_models()
        
        if self.available_models:
            names = [m['name'] for m in self.available_models]
            self.model_dropdown.values = names
            self.model_dropdown.set(names[0])
            self.model_dropdown.label.config(text=names[0])  # Force label update
            self.current_model_info = self.available_models[0]
            
            self._update_model_info()
        else:
            self.model_dropdown.label.config(text="No models found")
            self._show_error("No models found. Please check models_config.yaml")
    
    def _update_model_info(self):
        """Update model description"""
        if self.current_model_info:
            desc = self.current_model_info.get('description', '')
            size = self.current_model_info['size_mb']
            info = f"{desc} • {size:.1f} MB"
            self.model_desc.config(text=info)
    
    def _on_model_selected(self, name):
        """Model selected"""
        for model in self.available_models:
            if model['name'] == name:
                self.current_model_info = model
                self._update_model_info()
                if self.analyzer and not self.is_analyzing:
                    self.switch_btn.enable()
                break
    
    def _init_analyzer(self):
        """Initialize analyzer"""
        if not self.available_models:
            return
        
        def init():
            try:
                self.analyzer = VoiceAnalyzer(model_path=self.current_model_info['path'])
                self.root.after(0, self._on_ready)
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=init, daemon=True).start()
    
    def _on_ready(self):
        """Ready"""
        self._update_status("● Ready", COLORS['success'])
        self.select_btn.enable()
        
        # Enable Switch button if we have models
        if len(self.available_models) > 1:
            self.switch_btn.enable()
    
    def _switch_model(self):
        """Switch model"""
        if not self.current_model_info or self.is_analyzing:
            return
        
        self.switch_btn.disable()
        self._update_status("● Switching", COLORS['warning'])
        
        def switch():
            try:
                success = self.analyzer.load_model_weights(self.current_model_info['path'])
                self.root.after(0, lambda: self._on_switched(success))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=switch, daemon=True).start()
    
    def _on_switched(self, success):
        """Switched"""
        if success:
            self._update_status("● Ready", COLORS['success'])
            self._update_results(f"Switched to {self.current_model_info['name']}")
        self.switch_btn.disable()
    
    def _select_file(self):
        """Select file"""
        file = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                ("All files", "*.*")
            ]
        )
        
        if file:
            self.current_file = file
            name = os.path.basename(file)
            if len(name) > 30:
                name = name[:27] + "..."
            self.file_label.config(text=name, fg=COLORS['text'])
            self.analyze_btn.enable()
    
    def _analyze(self):
        """Analyze"""
        if not self.current_file or not self.analyzer or self.is_analyzing:
            return
        
        self.is_analyzing = True
        self.analyze_btn.disable()
        self.select_btn.disable()
        self.progress_label.config(text="⏳ Analyzing...")
        self.progress_label.pack(fill=tk.X, padx=15, pady=(0, 10))
        self._update_status("● Analyzing", COLORS['warning'])
        
        self._update_results("Analyzing audio file...")
        
        def analyze():
            try:
                results = self.analyzer.analyze(self.current_file)
                self.root.after(0, lambda: self._on_done(results))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def _on_done(self, results):
        """Done"""
        self.is_analyzing = False
        self.analyze_btn.enable()
        self.select_btn.enable()
        self.progress_label.pack_forget()
        
        filename = os.path.basename(self.current_file)
        pred = results['prediction']
        conf = results['confidence']
        male = results['male_probability']
        female = results['female_probability']
        
        result_text = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ANALYSIS COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: {filename}

PREDICTION: {pred.upper()}
Confidence: {conf:.1f}%

Detailed probabilities:
  • Male voice:   {male:.2f}%
  • Female voice: {female:.2f}%

Note: Analysis based on acoustic features only.
May not reflect speaker's gender identity."""
        
        self._update_results(result_text)
        self._update_status(f"● {pred} ({conf:.1f}%)", COLORS['success'])
    
    def _on_error(self, error):
        """Error"""
        self.is_analyzing = False
        self.analyze_btn.enable()
        self.select_btn.enable()
        self.progress_label.pack_forget()
        
        self._update_results(f"ERROR\n\n{error}\n\nPlease try another file.")
        self._update_status("● Error", COLORS['error'])
        messagebox.showerror("Analysis error", error)
    
    def _update_results(self, text):
        """Update results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', text)
        self.results_text.config(state=tk.DISABLED)
    
    def _update_status(self, text, color):
        """Update status"""
        self.status_badge.config(text=text, fg=color)
    
    def _show_error(self, msg):
        """Show error"""
        self._update_status("● Error", COLORS['error'])
        messagebox.showerror("Error", msg)


def main():
    # Check and download models if needed
    try:
        from download_models import check_models, download_models
        missing = check_models()
        if missing:
            print("Models not found. Downloading from GitHub Releases...")
            print("This is a one-time download (~4.4 MB total)")
            print()
            if not download_models():
                messagebox.showerror(
                    "Download Failed",
                    "Failed to download models from GitHub Releases.\n\n"
                    "Please check your internet connection and try again,\n"
                    "or download manually from:\n"
                    "https://github.com/Honey181/voicemd/releases"
                )
                return
            print()
    except Exception as e:
        print(f"Model check failed: {e}")
    
    root = tk.Tk()
    app = ModernVoiceMDApp(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
