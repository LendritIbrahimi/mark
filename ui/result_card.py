"""ResultCard widget -- displays the final result of a completed run."""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import BORDER, FG, FG_DIM, FG_GREEN, MONO, PAD, SURFACE, WIDTH


class ResultCard(ctk.CTkFrame):

    def __init__(self, parent: ctk.CTkBaseClass, result: str) -> None:
        super().__init__(
            parent, corner_radius=0,
            fg_color=SURFACE, border_width=1,
            border_color=FG_GREEN,
        )

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=PAD, pady=(3, 0))

        ctk.CTkLabel(
            hdr, text="[result]",
            font=ctk.CTkFont(family=MONO, size=14),
            text_color=FG, anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            hdr, text="[ ok]",
            font=ctk.CTkFont(family=MONO, size=14),
            text_color=FG_GREEN, anchor="e",
        ).pack(side="right")

        ctk.CTkLabel(
            self, text=result,
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG_DIM, anchor="w", justify="left",
            wraplength=WIDTH - 50,
        ).pack(fill="x", padx=PAD, pady=(1, 3))
