from collections import Counter

# ...existing code with sets...

# Count frequencies and sort
summary_freq = sorted(Counter(sum(summary_terms, [])).items(), key=lambda x: x[1], reverse=True)
summary_lemma_freq = sorted(Counter(sum(summary_lemma_terms, [])).items(), key=lambda x: x[1], reverse=True)
summary_expanded_freq = sorted(Counter(sum(summary_expanded_terms, [])).items(), key=lambda x: x[1], reverse=True)

# Print results (optional)
for term, freq in summary_freq:
    print(f"{term}: {freq}")

# Repeat for other sets if needed
