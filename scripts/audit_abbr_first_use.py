"""Show the first use of every listed abbreviation in the main text and whether it is expanded there."""
import re

D = "paper/ans"
main = open(f"{D}/geotrust_ans.tex", encoding="utf8").read()
full = re.sub(r"\\input\{([^}]*)\}", lambda m: open(f"{D}/{m.group(1)}.tex", encoding="utf8").read(), main)
body = full[full.index("\\maketitle"):full.index("\\section*{List of abbreviations}")]
body += full[full.index("\\begin{appendices}"):full.index("\\bibliography")]
lst = full[full.index("\\section*{List of abbreviations}"):full.index("\\backmatter")]
items = dict(re.findall(r"\\item\[([^\]]+)\] ([^\n]+)", lst))
clean = re.sub(r"\\(label|ref|citep|citet|includegraphics)(\[[^\]]*\])?\{[^}]*\}", "", body)
for k, v in sorted(items.items(), key=lambda kv: kv[0].upper()):
    keys = k.split("/")
    m = min((mm for kk in keys for mm in [re.search(r"(?<![A-Za-z])" + re.escape(kk) + r"(?![a-z])", clean)] if mm),
            key=lambda x: x.start(), default=None)
    if not m:
        print(f"{k:9s} NOT USED in text"); continue
    ctx = clean[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
    words = [w.lower() for w in re.findall(r"[a-z]{4,}", v.lower())][:2]
    ok = all(w in ctx.lower() for w in words) or f"({k})" in ctx
    print(f"{k:9s} {'ok ' if ok else '-- '} ...{ctx[-110:]}")
