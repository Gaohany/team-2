from string import Template

t = Template(r"""
\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[
    xbar stacked,
    legend style={
    legend columns=4,
        at={(xticklabel cs:0.5)},
        anchor=north,
        draw=none
    },
    ytick=data,
    axis y line*=none,
    axis x line*=bottom,
    tick label style={font=\footnotesize},
    legend style={font=\footnotesize},
    label style={font=\footnotesize},
    xtick={0,0.2,...,1.0},
    xmajorgrids = true,
    width=.9\linewidth,
    bar width=2mm,
    yticklabels={${labels}},
    xmin=0,
    xmax=1,
    area legend,
    enlarge y limits={abs=0.625},
    % nodes near coords,
    % nodes near coords style={text=black, at ={(\pgfplotspointmeta,\pgfplotspointy)},anchor=west},
    visualization depends on=y \as \pgfplotspointy,
    every axis plot/.append style={fill},
    ${extra}
]
${plots}
\end{axis}  
\end{tikzpicture}
\caption{X}
\label{fig:stats}
\end{figure}
""")

# data = [
#     {"name":  "nc",  "value": 0.32432432432432434, "color": "blue"},
#     {"name":  "nbc",  "value": 0.5135135135135135, "color": "blue"},
#     {"name":  "kmnc",  "value": 0.35135135135135137, "color": "blue"},
#     {"name":  "fuzzing",  "value": 0.5135135135135135, "color": "blue_dark"},
#     {"name":  "simple",  "value": 0.5945945945945946, "color": "orange"},
#     {"name":  "overlay",  "value": 0.5675675675675675, "color": "orange"},
#     {"name":  "strange",  "value": 0.5675675675675675, "color": "orange"},
#     {"name":  "shadow",  "value": 0.6216216216216216, "color": "orange"},
#     {"name":  "meta_basic",  "value": 0.7297297297297297, "color": "orange_dark"},
#     {"name":  "medium",  "value": 0.6216216216216216, "color": "orange"},
#     {"name":  "red_circle",  "value": 0.40540540540540543, "color": "orange"},
#     {"name":  "blue_circle",  "value": 0.0, "color": "orange"},
#     {"name":  "triangle",  "value": 0.3783783783783784, "color": "orange"},
#     {"name":  "sign",  "value": 0.2702702702702703, "color": "orange"},
#     {"name":  "meta",  "value": 0.918918918918919, "color": "orange_dark"},
#     {"name":  "test",  "value": 0.4864864864864865, "color": "green"},
# ][::-1]
# extra = r"""extra x ticks = 0.4864864864864865,
#     extra x tick labels={},
#     extra x tick style={grid=major,major grid style={dashed, draw=cb_green}},"""

# data = [
#     {"name": "nc", "value": 109.83093102033749, "color": "blue"},
#     {"name": "nbc", "value": 112.5134883259618, "color": "blue"},
#     {"name": "kmnc", "value": 109.89139592370321, "color": "blue"},
#     {"name": "fuzzing", "value": 112.04093037095181, "color": "blue_dark"},
#     {"name": "simple", "value": 48.697441988213114, "color": "orange"},
#     {"name": "overlay", "value": 83.74395326126454, "color": "orange"},
#     {"name": "strange", "value": 64.42162766567496, "color": "orange"},
#     {"name": "shadow", "value": 54.396744439768234, "color": "orange"},
#     {"name": "meta_basic", "value": 59.9539531441622, "color": "orange_dark"},
#     {"name": "medium", "value": 61.51348837031875, "color": "orange"},
#     {"name": "red_circle", "value": 34.96860433179279, "color": "orange"},
#     {"name": "blue_circle", "value": 10.064650956974473, "color": "orange"},
#     {"name": "triangle", "value": 17.340930140295693, "color": "orange"},
#     {"name": "sign", "value": 1.4939534386923148, "color": "orange"},
#     {"name": "meta", "value": 60.21279073315998, "color": "orange_dark"},
#     {"name": "test", "value": 41.60488377061001, "color": "green"},
# ][::-1]
# extra = r"""extra x ticks = 41.60488377061001,
#     extra x tick labels={},
#     extra x tick style={grid=major,major grid style={dashed, draw=cb_green}},"""

data = [
    {"name": "nc", "ms_gt": 0.2894736842105263, "ms_rel": 0.05263157894736842, "color": "blue"},
    {"name": "nbc", "ms_gt": 0.39473684210526316, "ms_rel": 0.18421052631578946, "color": "blue"},
    {"name": "kmnc", "ms_gt" :0.34210526315789475, "ms_rel":0.05263157894736842, "color": "blue"},
    {"name": "fuzzing", "ms_gt": 0.39473684210526316, "ms_rel": 0.18421052631578946, "color": "blue_dark"},
    {"name": "simple", "ms_gt": 0.47368421052631576, "ms_rel": 0.3684210526315789, "color": "orange"},
    {"name": "overlay", "ms_gt" :0.6578947368421053, "ms_rel":0.5789473684210527, "color": "orange"},
    {"name": "strange", "ms_gt": 0.47368421052631576, "ms_rel": 0.42105263157894735, "color": "orange"},
    {"name": "shadow", "ms_gt" :0.5526315789473685, "ms_rel":0.5263157894736842, "color": "orange"},
    {"name": "meta", "ms_gt": 0.7105263157894737, "ms_rel": 0.6842105263157895, "color": "orange_dark"},
    {"name": "test", "ms_gt" :0.47368421052631576, "ms_rel":0.47368421052631576, "color": "green"},
][::-1]
extra = r"""extra x ticks = 0.47368421052631576,
    extra x tick labels={},
    extra x tick style={grid=major,major grid style={dashed, draw=cb_green}},"""



repl = '\\_'
labels_text = ",".join(f"{{{d['name'].replace('_', repl)}}}" for d in data)

plots = [
    (
        " ".join(f"({d['ms_rel']}, {j})" if i == j else f"(0, {j})" for j, _ in enumerate(data)),
        d['color']
    )
    for i, d in enumerate(data)
]

plots_text = "\n".join(f"\\addplot[cb_{c}] coordinates {{{t}}};" for t, c in plots)

results = t.substitute({"labels": labels_text, "plots": plots_text, "extra": extra})
print(results)
