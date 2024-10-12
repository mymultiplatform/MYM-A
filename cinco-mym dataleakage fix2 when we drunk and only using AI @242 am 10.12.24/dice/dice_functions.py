import tkinter as tk
import random
import time
import threading
import numpy as np

def setup_dice_ui(dice_frame, root, connect_button):
    title_label = tk.Label(dice_frame, text="3 FACE DICE", font=("Helvetica", 16))
    title_label.pack(pady=10)

    display_var = tk.StringVar()
    display_label = tk.Label(dice_frame, textvariable=display_var, font=("Helvetica", 24), width=5, height=2, relief="solid")
    display_label.pack(pady=20)

    message_label = tk.Label(dice_frame, text="", font=("Helvetica", 14), fg="green")
    message_label.pack(pady=10)

    click_button = tk.Button(dice_frame, text="Click", font=("Helvetica", 14), 
                             command=lambda: on_button_click(root, display_var, click_button, message_label, connect_button))
    click_button.pack(pady=10)

    return display_var, click_button, message_label

def roll_dice(root, display_var, click_button, message_label, connect_button):
    def loading_animation():
        loading_text = ["", ".", "..", "..."]
        for _ in range(3):
            for text in loading_text:
                display_var.set(text)
                time.sleep(0.5)

    loading_animation()
    dice_result = random.randint(1, 3)
    display_var.set(dice_result)

    if dice_result == 3:
        message_label.config(text="Click Connect")
        root.after(0, connect_button.invoke)
    else:
        root.after(500, lambda: enable_button(click_button))

def on_button_click(root, display_var, click_button, message_label, connect_button):
    threading.Thread(target=lambda: roll_dice(root, display_var, click_button, message_label, connect_button)).start()
    click_button.config(state="disabled")

def enable_button(click_button):
    click_button.config(state="normal")

def auto_roll(root, click_button, display_var):
    while True:
        time.sleep(10)
        if click_button["state"] == "normal":
            root.after(0, lambda: on_button_click(root, display_var, click_button, 
                                                  root.nametowidget('.!frame.!label3'), 
                                                  root.nametowidget('.!frame2.!button')))

def reset(message_label, display_var, click_button):
    message_label.config(text="")
    display_var.set("")
    enable_button(click_button)
