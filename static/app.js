const form = document.querySelector("#screen-form");
const messageField = document.querySelector("#message");
const count = document.querySelector("#character-count");
const button = document.querySelector("#screen-button");
const result = document.querySelector("#result");
const verdict = document.querySelector("#verdict");
const detail = document.querySelector("#result-detail");
const scoreValue = document.querySelector("#score-value");
const thresholdValue = document.querySelector("#threshold-value");
const scoreTrack = document.querySelector(".score-track");
const scoreFill = document.querySelector("#score-fill");
const generateButtons = document.querySelectorAll(".generate-button");
const generationStatus = document.querySelector("#generation-status");
const themeToggle = document.querySelector("#theme-toggle");
const themeColor = document.querySelector('meta[name="theme-color"]');

function applyTheme(theme) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  themeToggle.textContent = `Switch to ${isDark ? "light" : "dark"} mode`;
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeColor.content = isDark ? "#17221e" : "#f3f6f1";
}

try {
  applyTheme(localStorage.getItem("scamsheild-theme") || "light");
} catch (error) {
  console.warn("Could not load the saved appearance preference.", error);
  applyTheme("light");
}

themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  try {
    localStorage.setItem("scamsheild-theme", nextTheme);
  } catch (error) {
    console.warn("Could not save the appearance preference.", error);
  }
});

generateButtons.forEach((generateButton) => {
  generateButton.addEventListener("click", async () => {
    generateButtons.forEach((button) => { button.disabled = true; });
    result.hidden = true;
    generationStatus.textContent = "Creating and independently screening a fictional test example...";
    try {
      const response = await fetch("/api/generate-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_type: generateButton.dataset.sampleType }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "A test example could not be generated.");
      messageField.value = data.message;
      messageField.dispatchEvent(new Event("input", { bubbles: true }));
      result.hidden = false;
      result.dataset.kind = data.prediction;
      verdict.textContent = data.detector_verdict === "fraudulent" ? "Likely spam" : "No spam detected";
      detail.textContent = `Generated sample label: ${data.ground_truth}. Independent detector decision: ${data.detector_verdict}. ${
        data.matches_ground_truth ? "The detector agrees with the synthetic label." : "The detector does not match the synthetic label."
      }`;
      scoreValue.textContent = `${Math.round(data.spam_score * 100)}%`;
      thresholdValue.textContent = `${Math.round(data.spam_threshold * 100)}%`;
      scoreTrack.setAttribute("aria-valuenow", String(Math.round(data.spam_score * 100)));
      scoreFill.style.width = `${Math.round(data.spam_score * 100)}%`;
      generationStatus.textContent = data.notice;
    } catch (error) {
      generationStatus.textContent = error.message;
    } finally {
      generateButtons.forEach((button) => { button.disabled = false; });
    }
  });
});

messageField.addEventListener("input", () => {
  count.textContent = `${messageField.value.length.toLocaleString()} / 5,000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "Screening...";
  result.hidden = false;
  result.dataset.kind = "pending";
  verdict.textContent = "Checking the message";
  detail.textContent = "Comparing its text with patterns learned from SMS spam examples.";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: messageField.value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The message could not be screened.");

    const percentage = Math.round(data.spam_score * 100);
    result.dataset.kind = data.prediction;
    verdict.textContent = data.verdict;
    detail.textContent = data.prediction === "spam"
      ? "This message has patterns associated with SMS spam. Smishing (phishing by text) is grouped into the spam class; it is not a separate prediction."
      : "The model did not find strong patterns associated with SMS spam or smishing. A clean result does not guarantee that a message is safe.";
    scoreValue.textContent = `${percentage}%`;
    thresholdValue.textContent = `${Math.round(data.spam_threshold * 100)}%`;
    scoreTrack.setAttribute("aria-valuenow", String(percentage));
    scoreFill.style.width = `${percentage}%`;
  } catch (error) {
    result.dataset.kind = "error";
    verdict.textContent = "Could not screen this message";
    detail.textContent = error.message;
    scoreValue.textContent = "--";
    scoreTrack.setAttribute("aria-valuenow", "0");
    scoreFill.style.width = "0%";
  } finally {
    button.disabled = false;
    button.innerHTML = 'Screen message <span aria-hidden="true">&#8594;</span>';
  }
});