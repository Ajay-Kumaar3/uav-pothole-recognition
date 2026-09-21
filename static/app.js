document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentMode = "city"; // "city" | "single"
  let citySurveyData = null;
  let selectedRelPath = null;
  let uploadedFile = null;
  let currentInspectionData = null;
  let activeTab = "annotated";
  let currentRoadBase64 = null;
  let currentRoadName = null;

  // Mode Navigation Elements
  const modeCityBtn = document.getElementById("modeCityBtn");
  const modeSingleBtn = document.getElementById("modeSingleBtn");
  const cityView = document.getElementById("cityView");
  const singleView = document.getElementById("singleView");

  // City Mode Elements
  const topRoadTitle = document.getElementById("topRoadTitle");
  const topRoadDesc = document.getElementById("topRoadDesc");
  const runCitySurveyBtn = document.getElementById("runCitySurveyBtn");
  const uploadMultiBtn = document.getElementById("uploadMultiBtn");
  const multiFileInput = document.getElementById("multiFileInput");
  const exportCityCsvBtn = document.getElementById("exportCityCsvBtn");

  const totalRoadsCount = document.getElementById("totalRoadsCount");
  const emergencyRoadsCount = document.getElementById("emergencyRoadsCount");
  const scheduledRoadsCount = document.getElementById("scheduledRoadsCount");
  const cityPotholesCount = document.getElementById("cityPotholesCount");
  const cityCracksCount = document.getElementById("cityCracksCount");
  const rankedRoadsBody = document.getElementById("rankedRoadsBody");

  // Single Road Mode Elements
  const sampleGrid = document.getElementById("sampleGrid");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const confSlider = document.getElementById("confSlider");
  const confVal = document.getElementById("confVal");
  const claheToggle = document.getElementById("claheToggle");
  const runBtn = document.getElementById("runBtn");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const mainImage = document.getElementById("mainImage");
  const emptyState = document.getElementById("emptyState");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const imageMeta = document.getElementById("imageMeta");

  const mpsValue = document.getElementById("mpsValue");
  const mpsBadge = document.getElementById("mpsBadge");
  const totalDefects = document.getElementById("totalDefects");
  const potholeCount = document.getElementById("potholeCount");
  const crackCount = document.getElementById("crackCount");

  const minorCount = document.getElementById("minorCount");
  const modCount = document.getElementById("modCount");
  const severeCount = document.getElementById("severeCount");
  const minorBar = document.getElementById("minorBar");
  const modBar = document.getElementById("modBar");
  const severeBar = document.getElementById("severeBar");

  const defectsTableBody = document.getElementById("defectsTableBody");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");

  // ==========================================
  // MODE SWITCHING LOGIC
  // ==========================================
  function switchMode(mode) {
    currentMode = mode;
    if (mode === "city") {
      cityView.style.display = "flex";
      singleView.style.display = "none";
      modeCityBtn.classList.add("active");
      modeSingleBtn.classList.remove("active");
    } else {
      cityView.style.display = "none";
      singleView.style.display = "grid";
      modeSingleBtn.classList.add("active");
      modeCityBtn.classList.remove("active");
    }
  }

  modeCityBtn.addEventListener("click", () => switchMode("city"));
  modeSingleBtn.addEventListener("click", () => switchMode("single"));

  // ==========================================
  // CITY-WIDE ROAD REPAIR PRIORITY (MODE 1)
  // ==========================================
  async function fetchCitySurvey(files = null) {
    rankedRoadsBody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center" style="padding: 40px;">
          <div class="spinner" style="margin: 0 auto 12px;"></div>
          <p style="color: var(--text-primary); font-weight:600;">Analyzing and Ranking Surveyed City Roads with AI...</p>
          <small style="color: var(--text-muted);">Applying CLAHE contrast enhancement & YOLOv11 defect recognition across all sectors.</small>
        </td>
      </tr>
    `;

    topRoadTitle.textContent = "AI Drone Fleet Auditing City Roads...";
    topRoadDesc.textContent = "Calculating Maintenance Priority Scores (MPS) to determine exact asphalt truck dispatch sequence.";

    const formData = new FormData();
    formData.append("conf", "0.020");
    formData.append("apply_clahe", "true");

    if (files && files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }
    }

    try {
      const res = await fetch("/api/batch-survey", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      citySurveyData = data;

      // Update KPI Counter Cards
      totalRoadsCount.textContent = data.total_roads_surveyed;
      emergencyRoadsCount.textContent = data.emergency_roads_count;
      scheduledRoadsCount.textContent = data.scheduled_roads_count || 0;
      cityPotholesCount.textContent = data.total_city_potholes;
      cityCracksCount.textContent = data.total_city_cracks;

      // Update Hero Banner with Rank #1
      if (data.ranked_roads && data.ranked_roads.length > 0) {
        const topRoad = data.ranked_roads[0];
        topRoadTitle.textContent = `🚨 PRIORITY #1: ${topRoad.road_name}`;
        topRoadDesc.textContent = `Critical Priority Score: ${topRoad.mps} (Contains ${topRoad.potholes} Potholes, ${topRoad.cracks} Cracks). Dispatch repair crews to this sector immediately.`;
      }

      // Render Leaderboard Rows
      renderLeaderboard(data.ranked_roads);

    } catch (err) {
      console.error("Failed to execute city survey:", err);
      rankedRoadsBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center" style="padding: 30px; color: var(--accent-red);">
            Failed to connect to AI survey backend. Please verify that the Flask server is running.
          </td>
        </tr>
      `;
    }
  }

  function renderLeaderboard(roads) {
    if (!roads || roads.length === 0) {
      rankedRoadsBody.innerHTML = `
        <tr><td colspan="8" class="text-center" style="padding:30px; color:var(--text-muted);">No road survey data available.</td></tr>
      `;
      return;
    }

    rankedRoadsBody.innerHTML = roads.map((r, idx) => {
      let rankClass = "rank-other";
      if (r.rank === 1) rankClass = "rank-1";
      else if (r.rank === 2) rankClass = "rank-2";
      else if (r.rank === 3) rankClass = "rank-3";

      let statusBadge = `<span class="status-badge badge-routine">🟢 ROUTINE (Fix Later)</span>`;
      let mpsColor = "var(--accent-green)";
      if (r.status === "EMERGENCY DISPATCH") {
        statusBadge = `<span class="status-badge badge-emergency">🚨 EMERGENCY (Fix 1st)</span>`;
        mpsColor = "var(--accent-red)";
      } else if (r.status === "SCHEDULED REPAIR") {
        statusBadge = `<span class="status-badge badge-scheduled">🟡 SCHEDULED (Fix 2nd)</span>`;
        mpsColor = "var(--accent-orange)";
      }

      return `
        <tr>
          <td>
            <span class="rank-pill ${rankClass}">#${r.rank}</span>
          </td>
          <td>
            <img class="road-thumb" src="${r.thumbnail}" alt="${r.road_name}" title="Click 'Deep Inspect Road' to view full resolution">
          </td>
          <td class="road-name-cell">
            <strong>${r.road_name}</strong>
            <small>${r.total_defects} Total Defects (${r.potholes} Potholes, ${r.cracks} Cracks)</small>
          </td>
          <td>
            <span class="mps-score" style="color: ${mpsColor};">${r.mps}</span>
          </td>
          <td>
            ${statusBadge}
          </td>
          <td>
            <span style="font-weight:700; color:var(--accent-red); font-family:var(--font-mono);">${r.potholes}</span>
          </td>
          <td>
            <span style="font-weight:700; color:var(--accent-blue); font-family:var(--font-mono);">${r.cracks}</span>
          </td>
          <td>
            <button class="btn-inspect" data-rank="${r.rank}">
              Deep Inspect &rarr;
            </button>
          </td>
        </tr>
      `;
    }).join("");

    // Wire "Deep Inspect" buttons to switch to Single Road View
    document.querySelectorAll(".btn-inspect").forEach(btn => {
      btn.addEventListener("click", () => {
        const rank = parseInt(btn.dataset.rank);
        const road = roads.find(r => r.rank === rank);
        if (road) {
          inspectSpecificRoad(road);
        }
      });
    });
  }

  function inspectSpecificRoad(road) {
    // Populate Single Road View with this road's AI data
    currentInspectionData = {
      image_name: road.road_name,
      maintenance_priority_score: road.mps,
      total_defects: road.total_defects,
      pothole_count: road.potholes,
      crack_count: road.cracks,
      severity_breakdown: road.severity_breakdown,
      annotated_image: road.annotated_full,
      raw_image: road.raw_image,
      clahe_image: road.clahe_image,
      defects: road.defects || []
    };

    selectedRelPath = road.rel_path || null;
    currentRoadBase64 = road.raw_image || null;
    currentRoadName = road.road_name || null;
    uploadedFile = null;

    updateImageView();
    mainImage.style.display = "block";
    emptyState.style.display = "none";
    loadingOverlay.style.display = "none";
    imageMeta.textContent = `${road.road_name} (Priority Rank #${road.rank})`;

    // Telemetry updates
    mpsValue.textContent = road.mps.toFixed(1);
    const isEmergency = (road.mps >= 50.0 || road.potholes >= 3 || (road.severity_breakdown && road.severity_breakdown.Severe > 0));
    const isScheduled = (road.mps >= 15.0 || road.total_defects >= 1);

    if (isEmergency) {
      mpsBadge.textContent = "EMERGENCY (FIX 1ST)";
      mpsBadge.style.background = "rgba(239, 68, 68, 0.2)";
      mpsBadge.style.color = "var(--accent-red)";
    } else if (isScheduled) {
      mpsBadge.textContent = "SCHEDULED (FIX 2ND)";
      mpsBadge.style.background = "rgba(245, 158, 11, 0.2)";
      mpsBadge.style.color = "var(--accent-orange)";
    } else {
      mpsBadge.textContent = "ROUTINE (FIX LATER)";
      mpsBadge.style.background = "rgba(16, 185, 129, 0.2)";
      mpsBadge.style.color = "var(--accent-green)";
    }

    totalDefects.textContent = road.total_defects;
    potholeCount.textContent = road.potholes;
    crackCount.textContent = road.cracks;

    const minor = road.severity_breakdown.Minor || 0;
    const mod = road.severity_breakdown.Moderate || 0;
    const severe = road.severity_breakdown.Severe || 0;
    const total = Math.max(1, road.total_defects);

    minorCount.textContent = minor;
    modCount.textContent = mod;
    severeCount.textContent = severe;

    minorBar.style.width = `${(minor / total) * 100}%`;
    modBar.style.width = `${(mod / total) * 100}%`;
    severeBar.style.width = `${(severe / total) * 100}%`;

    renderTable(road.defects || []);

    downloadJsonBtn.disabled = false;
    downloadCsvBtn.disabled = false;

    // Switch view to Single Road
    switchMode("single");
  }

  // City Actions
  runCitySurveyBtn.addEventListener("click", () => fetchCitySurvey());

  uploadMultiBtn.addEventListener("click", () => multiFileInput.click());
  multiFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      fetchCitySurvey(e.target.files);
    }
  });

  exportCityCsvBtn.addEventListener("click", () => {
    if (!citySurveyData || !citySurveyData.ranked_roads) {
      alert("No survey data to export.");
      return;
    }
    let csv = "Priority_Rank,Road_Name,Priority_Score_MPS,Dispatch_Action,Urgency_Tier,Potholes_Count,Cracks_Count,Total_Defects\n";
    citySurveyData.ranked_roads.forEach(r => {
      csv += `${r.rank},"${r.road_name.replace(/"/g, '""')}",${r.mps},"${r.status}","${r.urgency_tier}",${r.potholes},${r.cracks},${r.total_defects}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `city_municipal_road_repair_priority_schedule_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ==========================================
  // SINGLE ROAD INSPECTION LOGIC (MODE 2)
  // ==========================================
  confSlider.addEventListener("input", () => {
    const val = parseFloat(confSlider.value);
    confVal.textContent = `${val.toFixed(3)} (${(val * 100).toFixed(1)}%)`;
  });

  async function fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      sampleGrid.innerHTML = "";

      data.samples.forEach((sample, idx) => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.textContent = `${sample.split.toUpperCase()} #${idx + 1}`;
        card.title = sample.filename;
        card.addEventListener("click", () => {
          document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          selectedRelPath = sample.rel_path;
          uploadedFile = null;
          imageMeta.textContent = `${sample.filename} (${sample.split})`;
          runInspection();
        });
        sampleGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load samples:", err);
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadedFile = e.target.files[0];
      selectedRelPath = null;
      document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
      imageMeta.textContent = `${uploadedFile.name} (Custom Upload)`;
      runInspection();
    }
  });

  runBtn.addEventListener("click", () => {
    if (!selectedRelPath && !uploadedFile && !currentInspectionData) {
      alert("Please select a road sample or upload an aerial photo first.");
      return;
    }
    runInspection();
  });

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      updateImageView();
    });
  });

  function updateImageView() {
    if (!currentInspectionData) return;
    if (activeTab === "annotated") {
      mainImage.src = currentInspectionData.annotated_image;
    } else if (activeTab === "clahe") {
      mainImage.src = currentInspectionData.clahe_image;
    } else if (activeTab === "raw") {
      mainImage.src = currentInspectionData.raw_image;
    }
  }

  async function runInspection() {
    loadingOverlay.style.display = "flex";
    emptyState.style.display = "none";
    mainImage.style.display = "none";

    const formData = new FormData();
    formData.append("conf", confSlider.value);
    formData.append("apply_clahe", claheToggle.checked ? "true" : "false");

    if (uploadedFile) {
      formData.append("file", uploadedFile);
    } else if (selectedRelPath) {
      formData.append("rel_path", selectedRelPath);
    } else if (currentRoadBase64) {
      formData.append("image_base64", currentRoadBase64);
      formData.append("image_name", currentRoadName || "road.jpg");
    }

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      currentInspectionData = data;

      updateImageView();
      mainImage.style.display = "block";
      loadingOverlay.style.display = "none";

      mpsValue.textContent = data.maintenance_priority_score.toFixed(1);
      const isEmergencySingle = (data.maintenance_priority_score >= 50.0 || data.pothole_count >= 3 || (data.severity_breakdown && data.severity_breakdown.Severe > 0));
      const isScheduledSingle = (data.maintenance_priority_score >= 15.0 || data.total_defects >= 1);

      if (isEmergencySingle) {
        mpsBadge.textContent = "EMERGENCY (FIX 1ST)";
        mpsBadge.style.background = "rgba(239, 68, 68, 0.2)";
        mpsBadge.style.color = "var(--accent-red)";
      } else if (isScheduledSingle) {
        mpsBadge.textContent = "SCHEDULED (FIX 2ND)";
        mpsBadge.style.background = "rgba(245, 158, 11, 0.2)";
        mpsBadge.style.color = "var(--accent-orange)";
      } else {
        mpsBadge.textContent = "ROUTINE (FIX LATER)";
        mpsBadge.style.background = "rgba(16, 185, 129, 0.2)";
        mpsBadge.style.color = "var(--accent-green)";
      }

      totalDefects.textContent = data.total_defects;
      potholeCount.textContent = data.pothole_count;
      crackCount.textContent = data.crack_count;

      const minor = data.severity_breakdown.Minor || 0;
      const mod = data.severity_breakdown.Moderate || 0;
      const severe = data.severity_breakdown.Severe || 0;
      const total = Math.max(1, data.total_defects);

      minorCount.textContent = minor;
      modCount.textContent = mod;
      severeCount.textContent = severe;

      minorBar.style.width = `${(minor / total) * 100}%`;
      modBar.style.width = `${(mod / total) * 100}%`;
      severeBar.style.width = `${(severe / total) * 100}%`;

      renderTable(data.defects);

      downloadJsonBtn.disabled = false;
      downloadCsvBtn.disabled = false;

    } catch (err) {
      console.error("Single road inspection error:", err);
      loadingOverlay.style.display = "none";
      emptyState.style.display = "block";
      alert("Error executing model inference. Check server logs.");
    }
  }

  function renderTable(defects) {
    if (!defects || defects.length === 0) {
      defectsTableBody.innerHTML = `
        <tr><td colspan="6" class="text-center" style="color:var(--text-muted);">No defects detected at confidence ${confSlider.value}. Try adjusting threshold.</td></tr>
      `;
      return;
    }

    defectsTableBody.innerHTML = defects.map((d, i) => {
      let badgeClass = "tag-green";
      if (d.severity === "Moderate") badgeClass = "tag-orange";
      if (d.severity === "Severe") badgeClass = "tag-red";

      return `
        <tr>
          <td>${i + 1}</td>
          <td><strong>${d.class_name}</strong></td>
          <td><code>${(d.confidence * 100).toFixed(1)}%</code></td>
          <td><span class="tier-tag ${badgeClass}">${d.severity}</span></td>
          <td><code>${(d.normalized_area * 100).toFixed(2)}% of frame</code></td>
          <td>${d.urgency}</td>
        </tr>
      `;
    }).join("");
  }

  downloadJsonBtn.addEventListener("click", () => {
    if (!currentInspectionData) return;
    const blob = new Blob([JSON.stringify(currentInspectionData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `road_inspection_${currentInspectionData.image_name}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  downloadCsvBtn.addEventListener("click", () => {
    if (!currentInspectionData || !currentInspectionData.defects) return;
    let csv = "Defect_ID,Class,Confidence,Severity,Normalized_Area,Urgency,BBox\n";
    currentInspectionData.defects.forEach((d, i) => {
      csv += `${i + 1},"${d.class_name}",${d.confidence},${d.severity},${d.normalized_area},"${d.urgency}","${d.bbox.join(" ")}"\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `road_defects_${currentInspectionData.image_name}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ==========================================
  // INITIAL LOAD
  // ==========================================
  fetchSamples();
  fetchCitySurvey(); // Auto-runs city survey on page load!
});
