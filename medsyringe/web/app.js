const $ = (id) => document.getElementById(id);
const names = {
  count: ["syringe"],
  scale: ["20-scale", "60-scale"],
  color: ["amber-brown", "orange"],
};
const modelName = {
  dino: "Grounding DINO Tiny",
  owl: "OWLv2 Base",
  "dino-trained": "Grounding DINO with local re-ranking",
  "dino-augmented": "Grounding DINO with augmented re-ranking",
};
const title = {syringe: "Syringes", scale: "Visual scale", color: "Barrel color"};
let report = null;
let objects = [];
let selected = null;
let candidate = null;
let selectedMarking = null;
let drawing = null;
let start = null;
let fillEditor = null;
let inventoryCheck = null;
let comparison = null;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

async function request(url, options) {
  const reply = await fetch(url, options);
  if (reply.ok) return reply.json();
  const message = (await reply.json().catch(() => ({}))).detail;
  throw new Error(typeof message === "string" ? message : "The request could not be completed.");
}

function post(url, value) {
  return request(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(value),
  });
}

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function path(subpath = "") {
  return "/api/observations/" + encodeURIComponent(report.id) + subpath;
}

function options(select, entries, placeholder) {
  const before = entries.some(([id]) => id === select.value) ? select.value : "";
  select.replaceChildren(new Option(placeholder, ""),
    ...entries.map(([value, label]) => new Option(label, value)));
  select.value = before;
}

async function refresh() {
  const recent = await request("/api/observations");
  const entries = recent.map((entry) => [
    entry.id, `${entry.name} · ${new Date(entry.created_at).toLocaleString("en")}`,
  ]);
  options($("recent"), entries, "Select a saved observation");
  options($("reference"), entries.filter(([id]) => id !== report?.id), "Select a previous observation");
}

function boxOverlay(bbox, label, cls, action) {
  const [left, top, right, bottom] = bbox;
  const box = node(action ? "button" : "div", undefined, `box ${cls}`);
  if (action) {
    box.type = "button";
    box.addEventListener("click", action);
    box.setAttribute("aria-label", `Inspect ${label}`);
  }
  box.dataset.label = label;
  Object.assign(box.style, {
    left: `${left*100}%`, top: `${top*100}%`,
    width: `${(right-left)*100}%`, height: `${(bottom-top)*100}%`,
  });
  return box;
}

function choose(id) {
  if (selected !== id && fillEditor) closeFillEditor();
  selected = id;
  selectedMarking = null;
  candidate = null;
  renderObjects();
}

function nextManualId() {
  let index = 1;
  while (objects.some((obj) => obj.id === `manual-${index}`)) index++;
  return `manual-${index}`;
}

function addCandidate(index) {
  if (objects.some(obj => obj.candidate_index === index)) return;
  const pred = report.original.below_threshold_candidates[index];
  const id = nextManualId();
  objects.push({id, bbox: [...pred.bbox], origin: "human", scale: null, color: null, candidate_index: index});
  choose(id);
  editsPending();
}

function nativeCropLink(bbox, label) {
  const query = new URLSearchParams();
  for (const coordinate of bbox) query.append("bbox", coordinate);
  const link = node("a", label, "ocr");
  link.href = path("/crop?") + query.toString();
  link.target = "_blank";
  link.rel = "noopener";
  return link;
}

function renderCandidates() {
  const below = report?.original.below_threshold_candidates || [];
  $("uncertain").hidden = !below.length;
  const previousExpansion = $("candidates").querySelector("details")?.open;
  const rows = below.map((pred, index) => {
    const row = node("div", undefined, "candidate-review");
    const originalScore = pred.original_score ?? pred.score;
    const label = `Inspect candidate ${index+1} · ${originalScore.toFixed(3)}`;
    const inspect = node("button", label, "secondary candidate-row");
    inspect.addEventListener("click", () => {
      if (fillEditor) closeFillEditor();
      candidate = index;
      selectedMarking = null;
      selected = null;
      renderObjects();
      $("source").scrollIntoView({block: "nearest"});
    });
    row.append(inspect);
    if (candidate === index && report.original.source_extension) {
      row.append(nativeCropLink(pred.bbox, "Inspect candidate at native resolution"));
    }
    if (objects.some(obj => obj.candidate_index === index)) {
      row.append(node("small", "Included in review"));
    } else if (candidate === index) {
      if (pred.review_reason) row.append(node("small", pred.review_reason));
      const add = node("button", `Add candidate ${index+1} to review`, "secondary");
      add.addEventListener("click", () => addCandidate(index));
      row.append(add);
    }
    return row;
  });
  if (rows.length > 2) {
    const more = node("details", undefined, "additional-candidates");
    more.open = Boolean(previousExpansion || (candidate != null && candidate >= 2));
    more.append(node("summary", `Show ${rows.length - 2} more candidate${rows.length === 3 ? "" : "s"}`), ...rows.slice(2));
    $("candidates").replaceChildren(...rows.slice(0, 2), more);
  } else {
    $("candidates").replaceChildren(...rows);
  }
}

function renderBoxes() {
  $("boxes").replaceChildren(...objects.map((obj) => boxOverlay(obj.bbox, obj.id,
    obj.id === selected ? "selected" : "", () => choose(obj.id))));
  if (selectedMarking) {
    const object = objects.find((item) => item.id === selectedMarking.id);
    if (selectedMarking.points) {
      renderFillPoints(selectedMarking.points, selectedMarking.labels);
    } else if (object) {
      const [x, y, r, b] = object.bbox;
      for (const token of selectedMarking.tokens || [selectedMarking.token]) {
        const [l0, t0, r0, b0] = token.crop_bbox;
        const box = [x+(r-x)*l0, y+(b-y)*t0, x+(r-x)*r0, y+(b-y)*b0];
        $("boxes").append(boxOverlay(box, token.text, "candidate"));
      }
    }
  }
  if (fillEditor) renderFillPoints(fillEditor.points, ["Lower mark", "Upper mark", "Piston"]);
  if (candidate !== null) {
    const data = report.original.below_threshold_candidates[candidate];
    $("boxes").append(boxOverlay(data.bbox, "Review candidate", "candidate"));
  }
}

function renderFillPoints(points, labels) {
  for (const [[x, y], label] of points.map((point, index) => [point, labels[index]])) {
    const marker = node("span", label, "fill-point");
    marker.style.left = `${x*100}%`;
    marker.style.top = `${y*100}%`;
    $("boxes").append(marker);
  }
}

function closeFillEditor() {
  fillEditor = null;
  $("boxes").classList.remove("marking-fill");
}

function beginFillEditor(obj) {
  drawing = null;
  start = null;
  $("boxes").classList.remove("drawing");
  selected = obj.id;
  selectedMarking = null;
  candidate = null;
  const previous = obj.manual_fill || obj.fill_level?.landmarks;
  const capacity = obj.scale ? parseInt(obj.scale, 10) :
    previous?.capacity_ml ?? obj.inscriptions?.capacity?.value ?? "";
  fillEditor = {id: obj.id, points: [], capacity, low: previous?.low.ml ?? "", high: previous?.high.ml ?? ""};
  $("boxes").classList.add("marking-fill");
  renderObjects();
  $("source").scrollIntoView({block: "nearest"});
}

function fillTools(obj) {
  const section = node("div", undefined, "fill-tools");
  if (fillEditor?.id !== obj.id) {
    const edit = node("button", "Mark fill on image", "secondary");
    edit.type = "button";
    edit.addEventListener("click", () => beginFillEditor(obj));
    section.append(edit);
    if (obj.fill_level?.status === "manual estimate" || obj.manual_fill) {
      const clear = node("button", "Clear fill", "secondary");
      clear.type = "button";
      clear.addEventListener("click", () => {
        obj.manual_fill = null;
        selectedMarking = null;
        renderObjects();
        editsPending();
      });
      section.append(clear);
    }
    return section;
  }
  const editor = fillEditor;
  const prompts = ["Click the lower-numbered graduation", "Click the higher-numbered graduation",
    "Click the piston edge facing the liquid, near the same side of the barrel"];
  const prompt = node("p", prompts[editor.points.length] || "Enter the printed values and apply the marks.");
  prompt.setAttribute("aria-live", "polite");
  section.append(prompt);
  const numbers = node("div", undefined, "fill-values");
  for (const [key, label] of [["capacity", "Capacity (mL)"], ["low", "Lower (mL)"], ["high", "Upper (mL)"]]) {
    const field = node("label", label);
    const input = node("input");
    Object.assign(input, {type: "number", min: 0, max: 250, step: "any", value: editor[key]});
    input.addEventListener("input", () => { editor[key] = input.value; });
    field.append(input);
    numbers.append(field);
  }
  section.append(numbers);
  const actions = node("div", undefined, "button-row");
  for (const [label, action, disabled] of [
    ["Undo point", () => { editor.points.pop(); renderObjects(); }, !editor.points.length],
    ["Cancel", () => { closeFillEditor(); renderObjects(); }, false],
    ["Apply marks", () => {
      const [capacity, low, high] = [editor.capacity, editor.low, editor.high]
        .map(value => value === "" ? NaN : Number(value));
      if (!(capacity <= 250 && 0 <= low && low < high && high <= capacity)) {
        status("Enter two increasing printed graduations within the nominal capacity.", true);
        return;
      }
      const [lowPoint, highPoint, piston] = editor.points;
      obj.manual_fill = {capacity_ml: capacity, low: {ml: low, point: lowPoint},
        high: {ml: high, point: highPoint}, piston};
      selectedMarking = {id: obj.id, points: editor.points, labels: [`${low} mL`, `${high} mL`, "Piston"]};
      closeFillEditor();
      renderObjects();
      editsPending();
    }, editor.points.length !== 3],
  ]) {
    const button = node("button", label, "secondary");
    button.type = "button";
    button.disabled = disabled;
    button.addEventListener("click", action);
    actions.append(button);
  }
  section.append(actions);
  return section;
}

function labelField(obj, field) {
  const label = node("label", title[field]);
  const select = node("select");
  options(select, names[field].map((x) => [x, x]), "Unknown");
  select.value = obj[field] || "";
  select.addEventListener("change", () => {
    obj[field] = select.value || null;
    editsPending();
  });
  label.append(select);
  return label;
}

function objectCard(obj) {
  const card = node("div", undefined, "object" + (selected === obj.id ? " active" : ""));
  const header = node("div", undefined, "object-head");
  const select = node("button", obj.id);
  select.addEventListener("click", () => choose(obj.id));
  const label = obj.score == null ? (obj.origin === "draft" ? "Draft label" : "Manual box") :
    `${obj.score.toFixed(3)} detection score`;
  header.append(select, node("small", label));
  const row = node("div", undefined, "field-row");
  row.append(labelField(obj, "scale"), labelField(obj, "color"));
  card.append(header, row);
  if (obj.candidate_index !== undefined) {
    card.append(node("small", `Reviewed from candidate ${obj.candidate_index+1}`, "ocr"));
  }
  if (report.original.source_extension) {
    card.append(nativeCropLink(obj.bbox, "Inspect native-resolution crop"));
  }
  if (obj.evidence) {
    const text = Object.values(obj.evidence).map((entry) => entry.reason);
    card.append(node("p", [...new Set(text)].join(". ") + ".", "ocr"));
  }
  if (obj.inscriptions && obj.inscriptions.status !== "not requested") {
    const section = node("details", undefined, "ocr");
    const capacity = obj.inscriptions.capacity;
    section.append(node("summary", capacity ? `Nominal capacity · ${capacity.value} ${capacity.unit}` :
      `Marking candidates · ${obj.inscriptions.status}`));
    for (const token of obj.inscriptions.candidates) {
      const evidence = node("button", `${token.text} · OCR ${Math.round(token.score*100)}% · ${token.rotation}°`, "secondary candidate-row");
      evidence.addEventListener("click", () => {
        selectedMarking = {id: obj.id, token};
        selected = obj.id;
        candidate = null;
        renderBoxes();
      });
      section.append(evidence);
    }
    card.append(section);
  }
  if (obj.liquid_appearance) {
    const measured = obj.liquid_appearance;
    const details = node("details", undefined, "ocr");
    const lightness = measured.lightness;
    details.append(node("summary", lightness ? `Liquid lightness · ${lightness.lstar} L* · ${lightness.category}` :
      `Liquid appearance · ${measured.status}`));
    if (lightness) {
      details.append(node("p", `Middle 50% of measured pixels · ${lightness.iqr.join("–")} L*`));
      const color = measured.apparent_color;
      if (color) {
        const row = node("p", undefined, "liquid-color");
        const swatch = node("span", undefined, "liquid-swatch");
        swatch.style.backgroundColor = `rgb(${color.srgb.join(",")})`;
        swatch.setAttribute("aria-hidden", "true");
        row.append(swatch, document.createTextNode(`Apparent color · a* ${color.lab[1]} · b* ${color.lab[2]} · C* ${color.chroma}`));
        details.append(row);
      }
      const evidence = node("button", "Show measured region", "secondary candidate-row");
      evidence.addEventListener("click", () => {
        selected = obj.id;
        selectedMarking = {id: obj.id, token: {crop_bbox: measured.region, text: "Measured liquid region"}};
        candidate = null;
        renderBoxes();
      });
      details.append(evidence);
    } else {
      details.append(node("p", measured.reason));
    }
    if (measured.opacity_reason) details.append(node("p", measured.opacity_reason));
    card.append(details);
  }
  if (obj.manual_fill) {
    card.append(node("p", "Fill landmarks ready to save.", "ocr"));
  } else if (obj.manual_fill === null) {
    card.append(node("p", "Fill estimate will be cleared on save.", "ocr"));
  } else if (obj.fill_level) {
    const fill = obj.fill_level;
    const details = node("details", undefined, "ocr");
    const measured = fill.status === "visual estimate";
    const manual = fill.status === "manual estimate";
    details.append(node("summary", measured || manual ?
      `${manual ? "Reviewed" : "Estimated"} fill · ≈ ${fill.percent}% · ${fill.ml} mL` : "Estimated fill · unknown"));
    if (manual) {
      const {low, high, piston} = fill.landmarks;
      const evidence = node("button", "Show reviewed landmarks", "secondary candidate-row");
      evidence.type = "button";
      evidence.addEventListener("click", () => {
        closeFillEditor();
        selected = obj.id;
        candidate = null;
        selectedMarking = {id: obj.id, points: [low.point, high.point, piston],
          labels: [`${low.ml} mL`, `${high.ml} mL`, "Piston"]};
        renderObjects();
      });
      details.append(evidence);
    } else if (measured) {
      details.append(node("p", `Visual range · ${fill.interval_ml.join("–")} mL (${fill.interval_percent.join("–")}%)`));
      const evidence = node("button", "Show piston and graduations", "secondary candidate-row");
      evidence.addEventListener("click", () => {
        const [[x1,y1],[x2,y2]] = fill.piston_line;
        const piston = {text: "Piston edge", crop_bbox: [Math.max(0, Math.min(x1,x2)-.014),
          Math.max(0, Math.min(y1,y2)-.014), Math.min(1, Math.max(x1,x2)+.014), Math.min(1, Math.max(y1,y2)+.014)]};
        selected = obj.id;
        selectedMarking = {id: obj.id, tokens: [piston, ...fill.graduations.map(mark => ({
          crop_bbox: mark.crop_bbox, text: `${mark.ml} mL mark`,
        }))]};
        candidate = null;
        renderBoxes();
      });
      details.append(evidence);
    } else {
      details.append(node("p", fill.reason));
    }
    card.append(details);
  }
  card.append(fillTools(obj));
  const transcription = node("label", "Verified inscription");
  const input = node("input");
  input.maxLength = 320;
  input.value = obj.transcription || "";
  input.addEventListener("input", () => {
    obj.transcription = input.value;
    editsPending();
  });
  transcription.append(input);
  card.append(transcription);
  const remove = node("button", "Remove object", "remove");
  remove.addEventListener("click", () => {
    if (fillEditor?.id === obj.id) closeFillEditor();
    objects = objects.filter((item) => item.id !== obj.id);
    if (selected === obj.id) selected = null;
    renderObjects();
    editsPending();
  });
  card.append(remove);
  return card;
}

function renderObjects() {
  $("objects").replaceChildren(...objects.map(objectCard));
  $("total").textContent = objects.length;
  $("move").disabled = selected === null;
  renderBoxes();
  renderCandidates();
}

function clearChecks() {
  inventoryCheck = null;
  comparison = null;
  $("results").replaceChildren();
}

function editsPending() {
  clearChecks();
  status("Unsaved review changes. Save to use them in the report or reference checks.");
}

function show(data) {
  report = data;
  $("candidates").replaceChildren();
  objects = structuredClone(report.objects);
  selectedMarking = null;
  selected = null;
  candidate = null;
  drawing = null;
  start = null;
  closeFillEditor();
  $("boxes").classList.remove("drawing");
  clearChecks();
  $("export").disabled = false;
  $("empty").hidden = true;
  $("frame").hidden = false;
  $("source").src = path("/image");
  $("source").alt = report.name;
  $("revision").textContent = `Revision ${report.revision} · ${report.review_status}`;
  const metadata = report.original;
  $("summary").textContent = metadata.kind === "annotation"
    ? (metadata.flags.length ? metadata.flags.join(" · ") : "Draft annotation awaiting human review.")
    : (metadata.accepted_count ? `${modelName[metadata.model.name] || metadata.model.name} · score is uncalibrated`
      : "No detections were accepted. Review the source and any remaining candidates.");
  const checks = metadata.detection_checks;
  if (checks?.status === "degraded") {
    const warning = checks.review_needed ? "Verify this count before using the report." : "The image ran on the CPU.";
    $("summary").append(node("span", ` ${checks.unavailable.join(" and ")} unavailable. ${warning}`, "warning"));
  }
  $("review").hidden = false;
  $("approval").value = data.review_status === "approved" ? "approved" : "needs review";
  $("note").value = "";
  document.querySelectorAll("#inventory button, #comparison button").forEach((b) => b.disabled = false);
  renderObjects();
  status(report.name);
  refresh().catch((e) => status(e.message, true));
}

$("upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!$("file").files.length) {
    status("Choose an image first.");
    return;
  }
  $("analyze").disabled = true;
  status("Analyzing locally. The first image also loads the model.");
  try {
    const data = await request("/api/observations", {method: "POST", body: new FormData(event.target)});
    show(data);
  } catch (error) {
    status(error.message, true);
  } finally {
    $("analyze").disabled = false;
  }
});

$("file").addEventListener("change", () => {
  if ($("file").files.length) status(`Ready to analyze ${$("file").files[0].name}.`);
});

$("open-draft").addEventListener("click", async () => {
  if (!$("library").value) return;
  try {
    show(await request("/api/library/" + encodeURIComponent($("library").value), {method: "POST"}));
  } catch (error) {
    status(error.message, true);
  }
});
$("open-recent").addEventListener("click", async () => {
  if (!$("recent").value) return;
  try {
    show(await request("/api/observations/" + encodeURIComponent($("recent").value)));
  } catch (error) {
    status(error.message, true);
  }
});

function beginDrawing(id) {
  if (fillEditor) {
    closeFillEditor();
    renderObjects();
  }
  drawing = {id};
  start = null;
  $("boxes").classList.add("drawing");
  status("Drag across the image to mark the visible extent.");
}
$("insert").addEventListener("click", () => beginDrawing(null));
$("move").addEventListener("click", () => beginDrawing(selected));

function normalizedPoint(event) {
  const rect = $("boxes").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX-rect.left)/rect.width)),
    Math.max(0, Math.min(1, (event.clientY-rect.top)/rect.height)),
  ];
}

function drawBox(start, finish) {
  return [Math.min(start[0], finish[0]), Math.min(start[1], finish[1]),
    Math.max(start[0], finish[0]), Math.max(start[1], finish[1])];
}

$("boxes").addEventListener("pointerdown", (event) => {
  if (fillEditor) {
    const obj = objects.find(item => item.id === fillEditor.id);
    const [x, y] = normalizedPoint(event);
    const [l, t, r, b] = obj.bbox;
    if (x < l || x > r || y < t || y > b) {
      status("Place the fill landmark inside the selected syringe box.", true);
    } else if (fillEditor.points.length < 3) {
      fillEditor.points.push([x, y]);
      status("");
      renderObjects();
    }
    return;
  }
  if (!drawing) return;
  event.preventDefault();
  $("boxes").setPointerCapture(event.pointerId);
  start = normalizedPoint(event);
});
$("boxes").addEventListener("pointermove", (event) => {
  if (!start) return;
  renderBoxes();
  const preview = boxOverlay(drawBox(start, normalizedPoint(event)), "New box", "selected");
  $("boxes").append(preview);
});
$("boxes").addEventListener("pointerup", (event) => {
  if (!start || !drawing) return;
  const bbox = drawBox(start, normalizedPoint(event));
  if (bbox[2]-bbox[0] >= 0.005 && bbox[3]-bbox[1] >= 0.005) {
    if (drawing.id) {
      const changed = objects.find(obj => obj.id === drawing.id);
      changed.bbox = bbox;
      delete changed.manual_fill;
      delete changed.fill_level;
      delete changed.candidate_index;
      delete changed.reviewed_candidate;
    }
    else {
      selected = nextManualId();
      objects.push({id: selected, bbox, origin: "human", scale: null, color: null});
    }
    editsPending();
  } else status("The box is too small. Choose Draw box or Redraw selected to retry.");
  drawing = null;
  start = null;
  $("boxes").classList.remove("drawing");
  renderObjects();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && fillEditor) {
    closeFillEditor();
    renderObjects();
    return;
  }
  if (event.key !== "Escape" || !drawing) return;
  drawing = null;
  start = null;
  $("boxes").classList.remove("drawing");
  renderBoxes();
  status("Box drawing canceled.");
});

$("review").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (fillEditor) {
    status("Apply or cancel the fill landmarks before saving the review.", true);
    return;
  }
  try {
    show(await post(path("/review"), {
      revision: report.revision, reviewer: $("reviewer").value,
      note: $("note").value, status: $("approval").value, objects,
    }));
    status("Review saved. The original output remains in the report history.");
  } catch (error) {
    status(error.message, true);
  }
});

function referenceFields() {
  $("expected").replaceChildren(...names[$("mode").value].map((name) => {
    const label = node("label", title[name] || name);
    const input = node("input");
    Object.assign(input, {type: "number", min: 0, max: 250, required: true, value: 0, name});
    label.append(input);
    return label;
  }));
}

function renderChecks() {
  const rows = [];
  if (inventoryCheck) {
    for (const [key, expected] of Object.entries(inventoryCheck.expected)) {
      const count = inventoryCheck.observed[key];
      const detected = inventoryCheck.evidence[key].join(", ");
      const missing = inventoryCheck.missing[key];
      const extra = inventoryCheck.extra[key];
      rows.push(node("p", `${title[key] || key}: expected ${expected}, observed ${count}.` +
        (missing ? ` Missing ${missing}.` : "") + (extra ? ` Extra ${extra}.` : "") +
        (detected ? ` Evidence: ${detected}.` : "")));
    }
    if (inventoryCheck.unresolved.length) {
      rows.push(node("p", `Unresolved evidence: ${inventoryCheck.unresolved.join(", ")}. ` +
        `At least ${inventoryCheck.minimum_missing} expected items are missing.`));
    }
  }
  if (comparison) {
    rows.push(node("p", `Count changed from ${comparison.count_before} to ${comparison.count_after}. ${comparison.reason}`));
    for (const [field, counts] of Object.entries(comparison.attribute_count_changes)) {
      const changes = Object.entries(counts).filter(([, count]) => count !== 0)
        .map(([name, count]) => `${name} ${count > 0 ? "+" : ""}${count}`);
      if (changes.length) rows.push(node("p", `${title[field]}: ${changes.join(", ")}.`));
    }
  }
  $("results").replaceChildren(...rows);
}
$("mode").addEventListener("change", referenceFields);
$("inventory").addEventListener("submit", async (event) => {
  event.preventDefault();
  const counts = Object.fromEntries([...new FormData(event.target)]
    .map(([key, value]) => [key, Number(value)]));
  try {
    inventoryCheck = await post(path("/inventory"), {field: $("mode").value, counts});
    renderChecks();
  } catch (error) {
    status(error.message, true);
  }
});
$("comparison").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!$("reference").value) {
    status("Choose a reference observation.");
    return;
  }
  try {
    comparison = await request(path("/compare/" + encodeURIComponent($("reference").value)));
    renderChecks();
  } catch (error) {
    status(error.message, true);
  }
});
$("export").addEventListener("click", async () => {
  try {
    const reply = await fetch(path("/export"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        revision: report.revision,
        expected: inventoryCheck && {field: inventoryCheck.mode, counts: inventoryCheck.expected},
        reference_id: comparison?.previous_id,
        reference_revision: comparison?.previous_revision,
      }),
    });
    if (!reply.ok) throw new Error((await reply.json()).detail || "The export could not be created.");
    const address = URL.createObjectURL(await reply.blob());
    const link = node("a");
    link.href = address;
    link.download = `medsyringe-${report.id}.zip`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(address), 1000);
  } catch (error) {
    status(error.message, true);
  }
});

referenceFields();
refresh().catch((error) => status(error.message, true));
request("/api/library").then((items) => options($("library"),
  items.map(({id, flags}) => [id, `${id}${flags.length ? " · flagged" : ""}`]),
  "Select a source image")).catch((error) => status(error.message, true));
