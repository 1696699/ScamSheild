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
const feedbackPanel = document.querySelector("#feedback-panel");
const feedbackStatus = document.querySelector("#feedback-status");
const feedbackChoices = document.querySelectorAll(".feedback-choice");
const generateButton = document.querySelector("#generate-button");
const generationStatus = document.querySelector("#generation-status");
let currentPrediction = null;
let currentMessage = "";

generateButton.addEventListener("click", async () => {
  generateButton.disabled = true;
  result.hidden = true;
  feedbackPanel.hidden = true;
  currentPrediction = null;
  currentMessage = "";
  generationStatus.textContent = "Creating a fictional test example...";
  try {
    const response = await fetch("/api/generate-sample", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "A test example could not be generated.");
    messageField.value = data.message;
    messageField.dispatchEvent(new Event("input", { bubbles: true }));
    generationStatus.textContent = "Fictional sample generated. Review it, screen it, then rate the result.";
  } catch (error) {
    generationStatus.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
});

messageField.addEventListener("input", () => {
  count.textContent = `${messageField.value.length.toLocaleString()} / 5,000`;
  if (currentPrediction && messageField.value !== currentMessage) {
    result.hidden = true;
    feedbackPanel.hidden = true;
    currentPrediction = null;
    currentMessage = "";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "Screening...";
  result.hidden = false;
  feedbackPanel.hidden = true;
  feedbackStatus.textContent = "";
  currentPrediction = null;
  currentMessage = "";
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
    currentPrediction = data.prediction;
    currentMessage = messageField.value;
    verdict.textContent = data.verdict;
    detail.textContent = data.prediction === "spam"
      ? "This message has patterns the model associates with SMS spam."
      : "The model did not find strong patterns associated with SMS spam.";
    scoreValue.textContent = `${percentage}%`;
    thresholdValue.textContent = `${Math.round(data.spam_threshold * 100)}%`;
    scoreTrack.setAttribute("aria-valuenow", String(percentage));
    scoreFill.style.width = `${percentage}%`;
    feedbackChoices.forEach((choice) => { choice.disabled = false; });
    feedbackPanel.hidden = false;
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

feedbackChoices.forEach((choice) => {
  choice.addEventListener("click", async () => {
    if (!currentPrediction) return;

    feedbackChoices.forEach((button) => { button.disabled = true; });
    feedbackStatus.textContent = "Saving your feedback...";
    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actual_status: choice.dataset.status,
          prediction: currentPrediction,
          message: currentMessage,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Feedback could not be saved.");
      feedbackStatus.textContent = data.message;
    } catch (error) {
      feedbackStatus.textContent = error.message;
      feedbackChoices.forEach((button) => { button.disabled = false; });
    }
  });
});