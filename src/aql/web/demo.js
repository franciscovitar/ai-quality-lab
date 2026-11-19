"use strict";

// The page only presents the checked-in experimental snapshot. No model is run.
const categories = {
  supported: "RESPALDADA",
  contradicted: "CONTRADICCIÓN",
  non_answer: "NO RESPONDE",
  invalid_citation: "CITAS INVENTADAS",
  unsupported_command: "SIN RESPALDO",
};

const icons = {
  supported: "✓",
  contradicted: "×",
  non_answer: "?",
  invalid_citation: "#",
  unsupported_command: "!",
};

const byId = (id) => document.getElementById(id);

function selectCase(cases, buttons, index) {
  const item = cases[index];
  byId("case-id").textContent = `CASO ${item.id} / RESULTADO CONGELADO`;
  byId("case-title").textContent = item.title;
  byId("case-category").textContent = categories[item.category];
  byId("case-category").dataset.category = item.category;
  byId("case-question").textContent = item.question;
  byId("case-answer").textContent = item.model_answer;
  byId("case-observation").textContent = item.observation;
  byId("case-method").textContent = `TIPO DE EVIDENCIA · ${item.basis}`;
  byId("case-takeaway").textContent = item.takeaway;
  byId("case-progress").textContent = `CASO ${String(index + 1).padStart(2, "0")} DE ${String(cases.length).padStart(2, "0")}`;
  document.querySelector(".finding-box").dataset.category = item.category;

  const sourceNodes = item.source_passages.map((source) => {
    const entry = document.createElement("div");
    entry.className = "source-item";
    const label = document.createElement("span");
    label.className = "source-id";
    label.textContent = `[${source.id}] · FRAGMENTO EXACTO`;
    const quote = document.createElement("div");
    quote.className = "source-quote";
    quote.textContent = source.quote;
    entry.append(label, quote);
    return entry;
  });
  byId("case-sources").replaceChildren(...sourceNodes);

  for (const [buttonIndex, button] of buttons.entries()) {
    button.setAttribute("aria-pressed", String(buttonIndex === index));
  }
}

function renderCases(cases) {
  const buttons = cases.map((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Caso ${item.id}: ${item.title}`);

    const icon = document.createElement("span");
    icon.className = "case-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = icons[item.category];

    const content = document.createElement("span");
    content.className = "case-text";
    const title = document.createElement("strong");
    title.textContent = item.title;
    const subtitle = document.createElement("small");
    subtitle.textContent = `${item.id} · ${item.short_label}`;
    content.append(title, subtitle);

    const chevron = document.createElement("span");
    chevron.className = "case-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "›";
    button.append(icon, content, chevron);
    button.addEventListener("click", () => selectCase(cases, buttons, index));
    return button;
  });
  byId("case-list").replaceChildren(...buttons);
  selectCase(cases, buttons, 0);
}

async function initializeDemo() {
  try {
    const response = await fetch("/v1/demo/examples", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("Snapshot API unavailable");
    const data = await response.json();
    if (
      data.mode !== "frozen_synthetic_research_demo" ||
      !Array.isArray(data.examples) ||
      data.examples.length !== 5 ||
      data.examples.some((item) => !Object.hasOwn(categories, item.category))
    ) {
      throw new Error("Unexpected snapshot format");
    }
    renderCases(data.examples);
    // Link stays fixed to the known experiment, never derived from model text.
    byId("source-link").href =
      "https://github.com/franciscovitar/ai-quality-lab/actions/runs/44588063650";
    byId("explorer-layout").hidden = false;
  } catch (_error) {
    byId("demo-error").hidden = false;
  } finally {
    byId("demo-loading").hidden = true;
  }
}

initializeDemo();
