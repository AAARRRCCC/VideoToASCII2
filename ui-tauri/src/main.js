document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("ascii-form");
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");
  const outputArea = document.getElementById("output-area");
  const errorArea = document.getElementById("error-area");

  function resetUI() {
    progressContainer.style.display = "none";
    progressBar.style.width = "0%";
    progressText.textContent = "";
    outputArea.style.display = "none";
    errorArea.style.display = "none";
    outputArea.textContent = "";
    errorArea.textContent = "";
  }

  function validateForm(data) {
    if (!data.inputPath) return "Input video is required.";
    if (!data.outputPath) return "Output path is required.";
    if (data.width < 1 || data.width > 500) return "Width must be between 1 and 500.";
    if (data.height < 1 || data.height > 200) return "Height must be between 1 and 200.";
    if (data.fps < 1 || data.fps > 120) return "FPS must be between 1 and 120.";
    if (data.fontSize < 6 || data.fontSize > 48) return "Font size must be between 6 and 48.";
    if (data.batchSize < 1 || data.batchSize > 100) return "Batch size must be between 1 and 100.";
    if (data.scale < 1 || data.scale > 4) return "Scale must be between 1 and 4.";
    if (data.processes && (data.processes < 1 || data.processes > 64)) return "Processes must be between 1 and 64.";
    return null;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    resetUI();

    // Gather form data
    const inputFile = document.getElementById("input-path").files[0];
    const outputPath = document.getElementById("output-path").value.trim();
    const width = parseInt(document.getElementById("width").value, 10);
    const height = parseInt(document.getElementById("height").value, 10);
    const fps = parseInt(document.getElementById("fps").value, 10);
    const fontSize = parseInt(document.getElementById("font-size").value, 10);
    const tempDir = document.getElementById("temp-dir").value.trim();
    const processes = document.getElementById("processes").value.trim();
    const batchSize = parseInt(document.getElementById("batch-size").value, 10);
    const compare = document.getElementById("compare").checked;
    const mode = document.getElementById("mode").value;
    const scale = parseInt(document.getElementById("scale").value, 10);
    const profile = document.getElementById("profile").checked;

    const data = {
      inputPath: inputFile ? inputFile.name : "",
      outputPath,
      width,
      height,
      fps,
      fontSize,
      tempDir,
      processes: processes ? parseInt(processes, 10) : null,
      batchSize,
      compare,
      mode,
      scale,
      profile,
    };

    // Validate
    const error = validateForm(data);
    if (error) {
      errorArea.textContent = error;
      errorArea.style.display = "block";
      return;
    }

    // Simulate progress and output (replace with backend integration)
    progressContainer.style.display = "block";
    progressBar.style.width = "0%";
    progressText.textContent = "Starting...";

    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 18 + 7;
      if (progress > 100) progress = 100;
      progressBar.style.width = progress + "%";
      progressText.textContent = `Processing... ${Math.floor(progress)}%`;
      if (progress >= 100) {
        clearInterval(interval);
        progressText.textContent = "Done!";
        outputArea.textContent = "Conversion complete! (This is a demo. Backend integration coming soon.)";
        outputArea.style.display = "block";
      }
    }, 350);
  });
});
