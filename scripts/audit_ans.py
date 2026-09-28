"""Audit paper/ans against the Applied Network Science and Springer Nature template rules."""
import os, re

D = "paper/ans"
main = open(f"{D}/geotrust_ans.tex", encoding="utf8").read()
full = re.sub(r"\\input\{([^}]*)\}", lambda m: open(f"{D}/{m.group(1)}.tex", encoding="utf8").read(), main)
body = full[full.index("\\maketitle"):]
issues = []


def words(t):
    t = re.sub(r"\\(cite[tp]?|ref|label)\{[^}]*\}", "X", t)
    t = re.sub(r"\\[a-zA-Z]+\*?|[{}$\\]", " ", t)
    return len(t.split())


def captions(env):
    out = []
    for m in re.finditer(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", body, re.S):
        c = re.search(r"\\caption\{", m.group(0))
        if not c:
            continue
        i, depth = c.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(m.group(0)[i], 0); i += 1
        cap = m.group(0)[c.end():i - 1]
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        out.append((lab.group(1) if lab else "?", cap, m.group(0)))
    return out


# 1 abstract and keywords
a = full[full.index("\\abstract{") + 10:full.index("\\keywords")]
print("abstract words:", words(a), "(150-250)")
kw = full[full.index("\\keywords{") + 10:]; kw = kw[:kw.index("}")]
print("keywords:", len(kw.split(",")), "(3-10)")
if re.search(r"\\cite", a):
    issues.append("abstract contains a citation")

# 2 figure and table titles (first sentence <= 15 words), legends <= 300 words
for env in ("figure", "table"):
    for lab, cap, _ in captions(env):
        first = re.split(r"(?<=[.:])\s", cap.strip(), maxsplit=1)[0]
        nw, nt = words(first), words(cap)
        flag = "  <-- title > 15 words" if nw > 15 else ""
        flag += "  <-- legend > 300 words" if nt > 300 else ""
        print(f"{env:6s} {lab:22s} title {nw:3d} w, total {nt:3d} w{flag}")
        if nw > 15:
            issues.append(f"{env} {lab}: title has {nw} words (max 15)")

# 3 order of first citation of figures and tables
for kind, pfx in (("fig", "fig:"), ("tab", "tab:")):
    labels = [l for env in (("figure",) if kind == "fig" else ("table",)) for l, _, _ in captions(env)]
    aux = open(f"{D}/geotrust_ans.aux", encoding="utf8").read()
    num = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{(" + pfx + r"[^}]*)\}\{\{([^}]*)\}", aux)}
    first = []
    for m in re.finditer(r"\\ref\{(" + pfx + r"[^}]*)\}", body):
        if m.group(1) not in first:
            first.append(m.group(1))
    seq = [num.get(l, "?") for l in first]
    never = [l for l in labels if l not in first]
    print(f"{kind} first-mention order: {seq}; never cited: {never}")
    main_seq = [s for s in seq if s.isdigit()]
    if main_seq != sorted(main_seq, key=int):
        issues.append(f"{kind}s not cited in numerical order: {seq}")
    if never:
        issues.append(f"{kind}s never cited in the text: {never}")

# 4 images in subfolders
imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body)
sub = [i for i in imgs if "/" in i]
print("images:", len(imgs), "in subfolders:", len(sub))
if sub:
    issues.append(f"{len(sub)} images referenced from subfolders (template: keep them next to the .tex)")
for i in imgs:
    p = os.path.normpath(os.path.join(D, i))
    if os.path.exists(p) and os.path.getsize(p) > 10e6:
        issues.append(f"figure file > 10 MB: {i}")

# 5 layout commands, footnotes, colour, commas in numbers, URLs in text
for pat, what in ((r"\\(clearpage|newpage|pagebreak|vspace|smallskip|bigskip)\b", "layout command"),
                  (r"\\footnote\{", "footnote"), (r"\\textcolor|\\color\{", "coloured text"),
                  (r"\b\d{1,3},\d{3}\b", "comma in a number"), (r"\\url\{|https?://", "URL in text")):
    hits = [m.group(0) for m in re.finditer(pat, body[:body.index("\\bibliography")])]
    print(f"{what}: {len(hits)} {sorted(set(hits))[:6]}")
    if hits:
        issues.append(f"{what}: {len(hits)} occurrence(s)")

# 6 abbreviations used but not in the list
lst = full[full.index("List of abbreviations"):full.index("\\backmatter")]
listed = set(re.findall(r"\\item\[([^\]]+)\]", lst)) | {"BH", "GH", "TCs"}
scan = body[:body.index("List of abbreviations")] + body[body.index("\\begin{appendices}"):body.index("\\bibliography")]
txt = re.sub(r"\\(cite[tp]?|ref|label|includegraphics|begin|end|bmhead|citet|citep)(\[[^\]]*\])?\{[^}]*\}", " ", scan)
txt = re.sub(r"\$[^$]*\$", " ", txt)
cands = set(re.findall(r"\b([A-Z][A-Z0-9]{1,}[a-z]?(?:-[A-Z0-9]+)?)\b", txt))
ignore = {"GeoTrust", "RFC", "TCP", "II", "III", "IV", "ID", "OK", "AI", "LLM", "KB", "MHz", "STM32F405", "F407", "M4F", "UWBAD", "MPRs", "RUs", "UAVs", "HELLOs", "IEEE", "AES-CMAC", "STM32F407", "UTC", "CSV", "MB"} | {f"A{i}" for i in range(1, 16)} | {"A10a", "A10b"}
missing = sorted(c for c in cands - listed - ignore if not re.fullmatch(r"O\d|P\d|A\d+[ab]?|[A-Z]\d", c))
print("abbreviations not in list:", missing)
if missing:
    issues.append(f"abbreviations used but not listed/defined: {missing}")

# 7 declarations and required headings
for h in ("Availability of data and materials", "Competing interests", "Funding", "Authors' contributions",
          "Acknowledgements"):
    if h not in full:
        issues.append(f"missing declaration: {h}")
print("placeholders [..]:", len(re.findall(r"\[(?:To be|Given|Surname|Department|Institution|City|Country|corresponding|persistent|Authors to|to be inserted)", full)))
print("\n=== ISSUES ===")
for i in issues:
    print("-", i)
