/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, GroundingChunk, Type } from '@google/genai';

// --- Type Definitions ---
interface AnalysisResult {
  term: string;
  category: string;
  adGroup: string;
  positivePhrase: string;
  negativePhrase: string;
  competitorBrand:string;
  locationExclusion: string;
}

interface BusinessContext {
    location: string;
    competitors: string[];
    services: string[];
}

interface GroundingMetadata {
  groundingChunks?: GroundingChunk[];
}

// --- DOM Element References ---
const fileInput = document.getElementById('csv-file-input') as HTMLInputElement;
const urlInput = document.getElementById('website-url-input') as HTMLInputElement;
const manualLocationInput = document.getElementById('manual-location-input') as HTMLTextAreaElement;
const fileNameSpan = document.getElementById('file-name');
const analyzeButton = document.getElementById('analyze-button') as HTMLButtonElement;
const resultsContainer = document.getElementById('results-container');
const resultsTbody = document.getElementById('results-tbody') as HTMLTableSectionElement;
const copyCsvButton = document.getElementById('copy-csv-button') as HTMLButtonElement;
const sourcesSection = document.getElementById('sources-section');
const errorContainer = document.getElementById('error-container');
const buttonText = analyzeButton.querySelector('.button-text');
const spinner = analyzeButton.querySelector('.spinner') as HTMLElement;

let selectedFile: File | null = null;

// --- Gemini API Initialization ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- Event Listeners ---
fileInput.addEventListener('change', handleFileSelect);
analyzeButton.addEventListener('click', handleAnalyzeClick);
copyCsvButton.addEventListener('click', handleCopyCsvClick);

/**
 * Handles the file selection event.
 */
function handleFileSelect(): void {
  selectedFile = fileInput.files ? fileInput.files[0] : null;
  if (selectedFile) {
    fileNameSpan.textContent = selectedFile.name;
    analyzeButton.disabled = false;
  } else {
    fileNameSpan.textContent = 'Select Search Terms CSV';
    analyzeButton.disabled = true;
  }
}

/**
 * Fetches the business's context (location, competitors, services) from their website.
 * @param websiteUrl The URL of the business website.
 * @returns A promise that resolves to the business context.
 */
async function getBusinessContext(websiteUrl: string): Promise<BusinessContext> {
    const prompt = `
        Analyze the business at this website: ${websiteUrl}.
        Using a web search, find:
        1. The primary physical business location (e.g., "San Diego, CA").
        2. A list of its top 3-5 direct competitors.
        3. A concise list of the main products or services offered (e.g., ["emergency plumbing", "drain cleaning", "water heater repair"]).

        Return ONLY a single, minified JSON object with this exact structure: {"location": "string", "competitors": ["string"], "services": ["string"]}.
        Do not include any other text or markdown formatting.
    `;

    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: { parts: [{ text: prompt }] },
        config: {
            tools: [{ googleSearch: {} }],
        }
    });

    const cleanJsonText = response.text.replace(/```json\n?/, '').replace(/```/, '').trim();
    const parsed = JSON.parse(cleanJsonText);
    
    // Ensure the structure is correct, providing defaults if keys are missing.
    return {
        location: parsed.location || '',
        competitors: parsed.competitors || [],
        services: parsed.services || []
    };
}


/**
 * Main handler for the "Analyze" button click.
 */
async function handleAnalyzeClick(): Promise<void> {
  if (!selectedFile) {
    displayError('Please select a CSV file first.');
    return;
  }
  const websiteUrl = urlInput.value.trim();
  const manualLocation = manualLocationInput.value.trim();

  if (!websiteUrl && !manualLocation) {
      displayError("Please provide a Website URL or a manual Targeting Location.");
      return;
  }

  setLoadingState(true);
  clearResultsAndErrors();
  initializeResultDisplay();

  try {
    let context: BusinessContext = { location: '', competitors: [], services: [] };

    // Step 1: Get context.
    if (websiteUrl) {
        context = await getBusinessContext(websiteUrl);
    }
    // Prioritize manual location if provided.
    if (manualLocation) {
        context.location = manualLocation;
    }
    
    if (!context.location) {
        displayError("Could not determine a location. Please enter one manually.");
        setLoadingState(false);
        return;
    }


    // Step 2: Read the file and stream the analysis with the final context
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const csvContent = event.target?.result as string;
        const prompt = buildAnalysisPrompt(csvContent, context.location, context.competitors, context.services);

        const stream = await ai.models.generateContentStream({
          model: 'gemini-2.5-flash',
          contents: { parts: [{ text: prompt }] },
        });

        let buffer = '';
        for await (const chunk of stream) {
          buffer += chunk.text;
          let newlineIndex;
          while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
            const line = buffer.substring(0, newlineIndex).trim();
            buffer = buffer.substring(newlineIndex + 1);
            if (line) {
              tryParseAndDisplay(line);
            }
          }
        }
        if (buffer.trim()) {
          tryParseAndDisplay(buffer.trim());
        }
      } catch (error) {
         console.error(error);
         const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during streaming analysis.';
         displayError(`Analysis failed: ${errorMessage}`);
      } finally {
        setLoadingState(false);
      }
    };

    reader.onerror = () => {
      displayError('Failed to read the selected file.');
      setLoadingState(false);
    };
    reader.readAsText(selectedFile);

  } catch (error) {
    console.error(error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred while fetching business context.';
    displayError(`Analysis failed: ${errorMessage}`);
    setLoadingState(false);
  }
}

/**
 * Tries to parse a line of text as JSON and add it to the UI table.
 * @param line - The line of text to parse.
 */
function tryParseAndDisplay(line: string) {
  try {
    // Attempt to handle incomplete JSON objects gracefully
    const sanitizedLine = line.startsWith('{') && line.endsWith('}') ? line : '';
    if (!sanitizedLine) return;

    const item: AnalysisResult = JSON.parse(sanitizedLine);
    if (item.term && item.category) {
      addResultToTable(item);
    }
  } catch (e) {
    console.warn('Could not parse streaming JSON line:', line, e);
  }
}

/**
 * Builds the prompt for the Gemini API.
 * @param csvContent - The string content of the uploaded CSV.
 * @param location - The pre-determined business location.
 * @param competitors - The pre-determined list of competitors.
 * @param services - The pre-determined list of business services.
 * @returns The complete prompt string.
 */
function buildAnalysisPrompt(csvContent: string, location: string, competitors: string[], services: string[]): string {
    const servicesContext = services.length > 0
        ? `The business provides these specific services: ${services.join(', ')}.`
        : `The business's services were not provided; use general knowledge for the industry implied by the search terms.`;
  return `
    You are a high-speed Google Ads analysis engine. Your task is to analyze the following search terms from a CSV based on the provided business context. For each term, stream back a single, minified JSON object on its own line. Do not add any other text, explanations, or markdown. Stream only the line-delimited JSON.

    **Business Context:**
    - Primary Targeting Location: ${location}
    - Known Competitors: ${competitors.join(', ')}
    - ${servicesContext}

    **JSON Output Structure:**
    {"term": "string", "category": "string", "adGroup": "string", "positivePhrase": "string", "negativePhrase": "string", "competitorBrand": "string", "locationExclusion": "string"}

    **Categorization Rules:**
    1.  **'Positive'**: High-intent, relevant terms for the business that match its services and are within its location.
    2.  **'Negative'**:
        - Clearly irrelevant terms (e.g., 'jobs', 'free', 'DIY', 'how to').
        - Terms for products/services NOT offered by the business (based on the provided service list).
    3.  **'Competitor'**: Mentions a known competitor.
    4.  **'Generic'**: Broad terms that could be relevant but lack specific intent.

    **Field Definitions:**
    - "term": The original search term.
    - "category": Your classification based on the rules above.
    - "adGroup": Based on the term and business services, create a concise, thematic ad group name in Title Case (e.g., for "24 hour emergency plumber cost", the group could be "Emergency Plumbing"). If no clear theme, use "General".
    - "positivePhrase": If 'Positive', extract the valuable multi-word phrase. (e.g., for "24 hour emergency plumber cost", extract "24 hour emergency plumber"). Else, "".
    - "negativePhrase": If 'Negative', extract the word/phrase making it negative (e.g., for "plumber jobs", extract "jobs". For an un-offered service like "furnace installation", extract "furnace installation"). Else, "".
    - "competitorBrand": If 'Competitor', state the competitor's brand name. Else, "".
    - "locationExclusion": The Primary Location can be complex (cities, zips, radius). If a search term includes a specific place CLEARLY OUTSIDE this Primary Location, state that place here. Else, "".

    ### Search Term Data:
    \`\`\`csv
    ${csvContent}
    \`\`\`
  `;
}

/**
 * Toggles the loading state of the UI.
 * @param isLoading - Whether the app is in a loading state.
 */
function setLoadingState(isLoading: boolean): void {
  analyzeButton.disabled = isLoading;
  if (isLoading) {
    buttonText.textContent = 'Analyzing...';
    spinner.style.display = 'block';
  } else {
    buttonText.textContent = 'Analyze Terms';
    spinner.style.display = 'none';
  }
}

/**
 * Clears previous results and errors.
 */
function clearResultsAndErrors(): void {
  if (resultsContainer) resultsContainer.style.display = 'none';
  if (resultsTbody) resultsTbody.innerHTML = '';
  if (sourcesSection) sourcesSection.innerHTML = '';
  if (errorContainer) {
    errorContainer.style.display = 'none';
    errorContainer.textContent = '';
  }
}

/**
 * Displays an error message in the UI.
 * @param message - The error message to display.
 */
function displayError(message: string): void {
  errorContainer.textContent = message;
  errorContainer.style.display = 'block';
}

/**
 * Renders the initial empty state for result categories.
 */
function initializeResultDisplay(): void {
  if (resultsTbody) resultsTbody.innerHTML = '';
  if (resultsContainer) resultsContainer.style.display = 'block';
}

/**
 * Adds a new result to the table in the UI.
 * @param item - The analysis result object.
 */
function addResultToTable(item: AnalysisResult): void {
  if (!resultsTbody) return;

  const row = resultsTbody.insertRow();
  
  row.insertCell().textContent = item.term;
  row.insertCell().textContent = item.category;
  row.insertCell().textContent = item.adGroup;
  row.insertCell().textContent = item.positivePhrase;
  row.insertCell().textContent = item.negativePhrase;
  row.insertCell().textContent = item.competitorBrand;
  row.insertCell().textContent = item.locationExclusion;
  
  const rowCount = resultsTbody.rows.length;
  row.style.animationDelay = `${(rowCount % 50) * 20}ms`; // Stagger animation for large lists
}

/**
 * Handles the click event for the copy to CSV button.
 */
function handleCopyCsvClick(): void {
    const table = document.querySelector('.results-table') as HTMLTableElement;
    if (!table) return;

    const headers = Array.from(table.querySelectorAll('thead th')).map(th => `"${th.textContent}"`).join(',');
    const rows = Array.from(table.querySelectorAll('tbody tr')).map(row => 
        Array.from(row.querySelectorAll('td')).map(td => `"${(td.textContent || '').replace(/"/g, '""')}"`).join(',')
    );
    
    const csvContent = [headers, ...rows].join('\n');
    if (!csvContent) return;

    navigator.clipboard.writeText(csvContent).then(() => {
        copyCsvButton.textContent = 'Copied!';
        copyCsvButton.classList.add('copied');
        setTimeout(() => {
            copyCsvButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 0 24 24" width="18px" fill="currentColor"><path d="M0 0h24v24H0V0z" fill="none"/><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy as CSV`;
            copyCsvButton.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy CSV: ', err);
        copyCsvButton.textContent = 'Failed!';
         setTimeout(() => {
            copyCsvButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 0 24 24" width="18px" fill="currentColor"><path d="M0 0h24v24H0V0z" fill="none"/><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy as CSV`;
        }, 2000);
    });
}
