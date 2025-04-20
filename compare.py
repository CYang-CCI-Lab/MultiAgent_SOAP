async def _summarize_history(self, inplace: bool = True, message: str = "") -> bool | str:
    logger.warning("[%s] Starting iterative summarization …", self.__class__.__name__)

    async def _summarize_once(text: str) -> str | None:
        summary_prompt = (
            "Summarize the following message concisely, "
            "preserving all key facts and reasoning steps. "
            "Do not exceed 1000 words.\n\n"
            "<<<MESSAGE_START>>>\n"
            f"{text}\n"
            "<<<MESSAGE_END>>>"
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user",   "content": summary_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Summarization call failed: %s", e)
            return None

    if inplace:  
        if len(self.messages) < 3:
            logger.warning("Not enough messages to summarise.")
            return False

        def longest_idx() -> int:
            return max(
                (
                    (i, count_llama_tokens([m]))
                    for i, m in enumerate(self.messages)
                    if m["role"] != "system"
                ),
                key=lambda t: t[1],
            )[0]

        failures = 0
        while count_llama_tokens(self.messages) >= self.token_threshold:
            idx = longest_idx()
            summary = await _summarize_once(self.messages[idx]["content"])
            if summary is None:
                return False

            # sanity‑check: summary must be shorter
            if count_llama_tokens([{"role": "assistant", "content": summary}]) >= \
               count_llama_tokens([self.messages[idx]]):
                failures += 1
                if failures > 3:
                    logger.error("Summarization failed repeatedly. Aborting.")
                    return False
                continue

            self.messages[idx]["content"] = f"[Summary] {summary}"
            logger.info("Replaced longest message with summary (%d tokens total).",
                        count_llama_tokens(self.messages))
        return True

    # ── summarise an external `message` string ───────────────────────────────────────────
    if not message:
        logger.warning("No message provided for summarization.")
        return False
    
    summary = await _summarize_once(message)
    if summary is None:
        return False
    message = summary

    failures = 0
    while count_llama_tokens(message) >= self.token_threshold:
        summary = await _summarize_once(message)
        if summary is None:
            return False
        if count_llama_tokens(summary) >= count_llama_tokens(message):
            failures += 1
            if failures > 3:
                logger.error("Summarization failed repeatedly. Aborting.")
                return False
            continue
        message = summary     

    return message
