using Blink
using JSON3
using LinearAlgebra
using Base64

# Set BLAS threads for performance
BLAS.set_num_threads(Sys.CPU_THREADS)

# Include necessary modules
include("general_modules/channels.jl")
include("general_modules/mesh.jl")
include("swift/twobody.jl")
include("swift/MalflietTjon.jl")

using .channels
using .mesh
using .twobodybound
using .MalflietTjon

# Global variables
global calculation_running = false
global current_result = nothing
global current_ψtot = nothing
global current_ψ3 = nothing
global current_grid = nothing
global current_α = nothing

# HTML/CSS/JavaScript for the UI
const HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWIFT Nuclear Calculator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 350px 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .card-title {
            color: #667eea;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }

        .param-section {
            margin-bottom: 25px;
        }

        .section-label {
            color: #764ba2;
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 15px;
        }

        .param-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }

        .param-label {
            flex: 0 0 80px;
            color: #555;
            font-weight: 500;
        }

        .param-value {
            flex: 0 0 50px;
            text-align: center;
            color: #667eea;
            font-weight: 600;
            font-size: 1.1em;
        }

        input[type="range"] {
            flex: 1;
            margin: 0 10px;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            color: #555;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }

        select:hover {
            border-color: #667eea;
        }

        .toggle-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .toggle {
            position: relative;
            width: 60px;
            height: 30px;
            background: #ddd;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .toggle.active {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }

        .toggle-slider {
            position: absolute;
            top: 3px;
            left: 3px;
            width: 24px;
            height: 24px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s;
        }

        .toggle.active .toggle-slider {
            left: 33px;
        }

        .run-button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .run-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        .run-button:active {
            transform: translateY(0);
        }

        .run-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .status {
            text-align: center;
            padding: 10px;
            margin-top: 10px;
            border-radius: 8px;
            font-weight: 500;
        }

        .status.ready { background: #e8f5e9; color: #2e7d32; }
        .status.running { background: #fff3e0; color: #e65100; }
        .status.success { background: #e8f5e9; color: #2e7d32; }
        .status.error { background: #ffebee; color: #c62828; }

        .output-console {
            height: 500px;
            overflow-y: auto;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }

        .output-console::-webkit-scrollbar {
            width: 8px;
        }

        .output-console::-webkit-scrollbar-track {
            background: #2d2d2d;
        }

        .output-console::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }

        .result-item {
            padding: 15px;
            background: #f5f5f5;
            border-left: 4px solid #667eea;
            margin-bottom: 10px;
            border-radius: 5px;
        }

        .result-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .result-value {
            color: #333;
            font-size: 1.3em;
            font-weight: 600;
        }

        .viz-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .plot-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }

        .plot-canvas {
            width: 100%;
            height: 400px;
            background: #f9f9f9;
            border-radius: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 SWIFT Nuclear Structure Calculator</h1>
            <p>Scattering With Identical Faddeev Three-body solver</p>
        </div>

        <div class="main-grid">
            <!-- Left Panel: Parameters -->
            <div class="card">
                <div class="card-title">⚙️ Parameters</div>

                <div class="param-section">
                    <div class="section-label">Grid Parameters</div>
                    <div class="param-row">
                        <span class="param-label">nx:</span>
                        <input type="range" id="nx" min="10" max="30" step="5" value="20">
                        <span class="param-value" id="nx-val">20</span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">ny:</span>
                        <input type="range" id="ny" min="10" max="30" step="5" value="20">
                        <span class="param-value" id="ny-val">20</span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">nθ:</span>
                        <input type="range" id="ntheta" min="8" max="20" step="2" value="12">
                        <span class="param-value" id="ntheta-val">12</span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">xmax:</span>
                        <input type="range" id="xmax" min="10" max="24" step="2" value="16">
                        <span class="param-value" id="xmax-val">16</span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">ymax:</span>
                        <input type="range" id="ymax" min="10" max="24" step="2" value="16">
                        <span class="param-value" id="ymax-val">16</span>
                    </div>
                </div>

                <div class="param-section">
                    <div class="section-label">Basis Parameters</div>
                    <div class="param-row">
                        <span class="param-label">lmax:</span>
                        <input type="range" id="lmax" min="4" max="12" step="2" value="8">
                        <span class="param-value" id="lmax-val">8</span>
                    </div>
                    <div class="param-row">
                        <span class="param-label">λmax:</span>
                        <input type="range" id="lambdamax" min="10" max="30" step="5" value="20">
                        <span class="param-value" id="lambdamax-val">20</span>
                    </div>
                </div>

                <div class="param-section">
                    <div class="section-label">Potential</div>
                    <select id="potential">
                        <option value="AV18" selected>AV18</option>
                        <option value="AV14">AV14</option>
                        <option value="Nijmegen">Nijmegen</option>
                        <option value="MT_V">Malfliet-Tjon</option>
                    </select>
                </div>

                <div class="param-section">
                    <div class="toggle-container">
                        <span class="section-label">Include UIX</span>
                        <div class="toggle" id="uix-toggle">
                            <div class="toggle-slider"></div>
                        </div>
                    </div>
                </div>

                <button class="run-button" id="run-btn">▶ Run Calculation</button>
                <div class="status ready" id="status">Ready</div>
            </div>

            <!-- Middle Panel: Output -->
            <div class="card">
                <div class="card-title">📊 Calculation Output</div>
                <div class="output-console" id="output"></div>
            </div>

            <!-- Right Panel: Results -->
            <div class="card">
                <div class="card-title">📈 Results</div>
                <div id="results">
                    <div class="result-item">
                        <div class="result-label">Ground State Energy</div>
                        <div class="result-value" id="energy-gs">--</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Binding Energy</div>
                        <div class="result-value" id="energy-bind">--</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Iterations</div>
                        <div class="result-value" id="iterations">--</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Converged</div>
                        <div class="result-value" id="converged">--</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">λ Eigenvalue</div>
                        <div class="result-value" id="lambda-val">--</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom: Visualization -->
        <div class="viz-container">
            <div class="card-title">📉 Wavefunction Visualization</div>
            <div class="plot-area">
                <div id="plot1" class="plot-canvas" style="display:none; width:100%; height:650px;"></div>
                <div id="plot2" class="plot-canvas" style="display:none; width:100%; height:650px;"></div>
            </div>
        </div>
    </div>

    <script>
        // Plotly.js will be loaded by Julia via Blink.js()

        // Wait for Plotly to load, then define plot functions
        function waitForPlotly(callback, maxAttempts = 50) {
            var attempts = 0;
            var checkInterval = setInterval(function() {
                attempts++;
                if (typeof Plotly !== 'undefined') {
                    console.log('Plotly loaded successfully after', attempts, 'attempts');
                    clearInterval(checkInterval);
                    callback();
                } else if (attempts >= maxAttempts) {
                    console.error('Failed to load Plotly after', maxAttempts, 'attempts');
                    clearInterval(checkInterval);
                }
            }, 100);
        }

        // Global plot functions for Julia to call
        window.createPlot1 = function(z_data, x_data, y_data, channel_num) {
            console.log('createPlot1 called with data sizes:', z_data.length, 'x', z_data[0].length);

            if (typeof Plotly === 'undefined') {
                console.error('Plotly is not loaded yet! Waiting...');
                waitForPlotly(function() {
                    window.createPlot1(z_data, x_data, y_data, channel_num);
                });
                return 'waiting for Plotly...';
            }

            try {
                var trace = {
                    z: z_data,
                    x: x_data,
                    y: y_data,
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    colorbar: {title: 'Amplitude'}
                };
                var layout = {
                    title: 'Dominant Channel (Ch ' + channel_num + ') Wavefunction |ψ|²',
                    xaxis: {title: 'x (fm)'},
                    yaxis: {title: 'y (fm)'},
                    height: 600
                };
                Plotly.newPlot('plot1', [trace], layout);
                document.getElementById('plot1').style.display = 'block';
                console.log('Plot1 created successfully!');
                return 'success';
            } catch(e) {
                console.error('Error in createPlot1:', e);
                return 'error: ' + e.message;
            }
        };

        window.createPlot2 = function(x_data, prob_data) {
            console.log('createPlot2 called with data length:', x_data.length);

            if (typeof Plotly === 'undefined') {
                console.error('Plotly is not loaded yet! Waiting...');
                waitForPlotly(function() {
                    window.createPlot2(x_data, prob_data);
                });
                return 'waiting for Plotly...';
            }

            try {
                var trace = {
                    x: x_data,
                    y: prob_data,
                    type: 'scatter',
                    mode: 'lines',
                    line: {width: 3, color: 'rgb(102, 126, 234)'}
                };
                var layout = {
                    title: 'Total Radial Probability Distribution',
                    xaxis: {title: 'x (fm)'},
                    yaxis: {title: 'Probability'},
                    height: 600,
                    showlegend: false
                };
                Plotly.newPlot('plot2', [trace], layout);
                document.getElementById('plot2').style.display = 'block';
                console.log('Plot2 created successfully!');
                return 'success';
            } catch(e) {
                console.error('Error in createPlot2:', e);
                return 'error: ' + e.message;
            }
        };

        // Update slider values
        const sliders = ['nx', 'ny', 'ntheta', 'xmax', 'ymax', 'lmax', 'lambdamax'];
        sliders.forEach(id => {
            const slider = document.getElementById(id);
            const display = document.getElementById(id + '-val');
            slider.addEventListener('input', () => {
                display.textContent = slider.value;
            });
        });

        // Toggle switch
        let uixEnabled = false;
        document.getElementById('uix-toggle').addEventListener('click', function() {
            uixEnabled = !uixEnabled;
            this.classList.toggle('active');
        });

        // Output functions
        function appendOutput(text) {
            const output = document.getElementById('output');
            output.innerHTML += text + '<br>';
            output.scrollTop = output.scrollHeight;
        }

        function clearOutput() {
            document.getElementById('output').innerHTML = '';
        }

        function setStatus(text, className) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = 'status ' + className;
        }

        // Run calculation
        document.getElementById('run-btn').addEventListener('click', async function() {
            const btn = this;
            btn.disabled = true;
            clearOutput();
            setStatus('⏳ Running calculation...', 'running');

            const params = {
                nx: parseInt(document.getElementById('nx').value),
                ny: parseInt(document.getElementById('ny').value),
                ntheta: parseInt(document.getElementById('ntheta').value),
                xmax: parseFloat(document.getElementById('xmax').value),
                ymax: parseFloat(document.getElementById('ymax').value),
                lmax: parseInt(document.getElementById('lmax').value),
                lambdamax: parseInt(document.getElementById('lambdamax').value),
                potential: document.getElementById('potential').value,
                include_uix: uixEnabled
            };

            try {
                // Handler returns immediately; calculation runs in background
                await Blink.msg("run_calculation", params);
                // Julia will update status and re-enable button when done
            } catch(e) {
                appendOutput("<br><br>❌ JavaScript Error: " + e);
                setStatus('❌ Communication error', 'error');
                btn.disabled = false;  // Re-enable on communication errors only
            }
        });
    </script>
</body>
</html>
"""

function run_calculation_web(params::Dict)
    global win  # Need to access global window for @js_ macro

    try
        global calculation_running = true

        # Extract parameters
        nx = params["nx"]
        ny = params["ny"]
        nθ = params["ntheta"]
        xmax = Float64(params["xmax"])
        ymax = Float64(params["ymax"])
        lmax = params["lmax"]
        λmax = params["lambdamax"]
        potname = params["potential"]
        include_uix = params["include_uix"]

        # Fixed quantum numbers for 3H
        fermion = true
        Jtot = 0.5
        T = 0.5
        Parity = 1
        lmin = 0
        λmin = 0
        j2bmax = 1.0
        s1 = s2 = s3 = 0.5
        t1 = t2 = t3 = 0.5
        MT = -0.5
        alpha = 1.0
        E0_guess = -7.5
        E1_guess = -6.5

        # Send output to web UI
        function web_output(msg)
            fullmsg = msg * "<br>"
            @js_ win document.getElementById("output").innerHTML += $fullmsg
            @js_ win document.getElementById("output").scrollTop = document.getElementById("output").scrollHeight
        end

        web_output("="^70)
        web_output("    SWIFT THREE-BODY CALCULATION")
        web_output("="^70)
        web_output("Potential: $potname")
        web_output("Grid: nx=$nx, ny=$ny, nθ=$nθ")
        web_output("Basis: lmax=$lmax, λmax=$λmax")
        web_output("="^70)

        # Generate channels
        web_output("\nGenerating three-body channels...")
        α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
                s1, s2, s3, t1, t2, t3, MT, j2bmax)
        web_output("✓ Generated $(α.nchmax) channels")

        # Generate mesh
        web_output("\nGenerating numerical mesh...")
        grid = initialmesh(nθ, nx, ny, xmax, ymax, alpha)
        web_output("✓ Mesh initialized: $(nx)×$(ny) grid points")

        # Two-body bound state
        web_output("\nCalculating two-body bound state...")
        e2b, ψ2b = bound2b(grid, potname)
        web_output("✓ Two-body calculation complete")

        # Three-body calculation
        web_output("\nRunning Malfiet-Tjon solver...")
        web_output("Initial energy guesses: E0=$E0_guess MeV, E1=$E1_guess MeV")

        result, ψtot, ψ3 = malfiet_tjon_solve_optimized(
            α, grid, potname, e2b,
            E0=E0_guess,
            E1=E1_guess,
            tolerance=1e-6,
            max_iterations=30,
            verbose=false,
            include_uix=include_uix
        )

        web_output("\n" * "="^70)
        web_output("    CALCULATION COMPLETE")
        web_output("="^70)
        web_output("Ground state energy:  $(round(result.energy, digits=6)) MeV")
        web_output("Binding energy:       $(round(-result.energy, digits=6)) MeV")
        web_output("λ eigenvalue:         $(round(result.eigenvalue, digits=8))")
        web_output("Iterations:           $(result.iterations)")
        web_output("Converged:            $(result.converged ? "Yes" : "No")")
        web_output("="^70)

        # Update results display
        E_gs = result.energy
        E_binding = -E_gs
        n_iter = result.iterations
        converged = result.converged ? "✓ Yes" : "✗ No"

        # Format strings in Julia before sending to JavaScript
        energy_gs_text = string(round(E_gs, digits=4)) * " MeV"
        energy_bind_text = string(round(E_binding, digits=4)) * " MeV"
        lambda_text = string(round(result.eigenvalue, digits=6))

        @js_ win document.getElementById("energy-gs").textContent = $energy_gs_text
        @js_ win document.getElementById("energy-bind").textContent = $energy_bind_text
        @js_ win document.getElementById("iterations").textContent = $n_iter
        @js_ win document.getElementById("converged").textContent = $converged
        @js_ win document.getElementById("lambda-val").textContent = $lambda_text

        # Update status and re-enable button
        @js_ win document.getElementById("status").textContent = "✓ Calculation complete!"
        @js_ win document.getElementById("status").className = "status success"
        @js_ win document.getElementById("run-btn").disabled = false

        # Store results globally
        global current_result = result
        global current_ψtot = ψtot
        global current_ψ3 = ψ3
        global current_grid = grid
        global current_α = α

        # Generate wavefunction visualizations
        web_output("\nGenerating visualizations...")

        # Extract grid points
        x_points = grid.xi
        y_points = grid.yi

        # Find dominant channel (highest probability)
        probs = zeros(α.nchmax)
        for iα in 1:α.nchmax
            idx_start = (iα-1)*grid.nx*grid.ny + 1
            idx_end = iα*grid.nx*grid.ny
            probs[iα] = sum(abs2.(ψ3[idx_start:idx_end]))
        end
        dominant_ch = argmax(probs)

        # Extract dominant channel wavefunction (reshaped to 2D)
        idx_start = (dominant_ch-1)*grid.nx*grid.ny + 1
        idx_end = dominant_ch*grid.nx*grid.ny
        ψ_ch = reshape(abs2.(ψ3[idx_start:idx_end]), grid.ny, grid.nx)

        # Calculate radial probability distribution
        total_prob = zeros(grid.nx)
        for ix in 1:grid.nx
            for iα in 1:α.nchmax
                for iy in 1:grid.ny
                    idx = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
                    total_prob[ix] += abs2(ψ3[idx])
                end
            end
        end

        # Generate interactive plots using Plotly.js
        web_output("🔍 DEBUG: Starting visualization...")

        # Prepare data as Julia arrays
        z_data = collect(ψ_ch')
        x_data = collect(x_points)
        y_data = collect(y_points)
        prob_data = collect(total_prob)

        web_output("🔍 DEBUG: Data sizes - z: $(size(z_data)), x: $(length(x_data)), y: $(length(y_data)), prob: $(length(prob_data))")

        # Call JavaScript functions with data
        web_output("🔍 DEBUG: Calling createPlot1...")
        result1 = @js_ win window.createPlot1($z_data, $x_data, $y_data, $dominant_ch)
        web_output("🔍 DEBUG: createPlot1 result: $result1")

        web_output("🔍 DEBUG: Calling createPlot2...")
        result2 = @js_ win window.createPlot2($x_data, $prob_data)
        web_output("🔍 DEBUG: createPlot2 result: $result2")

        web_output("✓ Visualizations complete! Open developer tools (should open automatically) to see detailed logs.")

    catch e
        @js_ win document.getElementById("status").textContent = "❌ Error occurred"
        @js_ win document.getElementById("status").className = "status error"

        error_msg = sprint(showerror, e)
        full_error = "<br>❌ ERROR:<br>" * error_msg
        @js_ win document.getElementById("output").innerHTML += $full_error

        println("Error: ", e)
        println(stacktrace(catch_backtrace()))
    finally
        global calculation_running = false
    end
end

function main()
    println("="^70)
    println("    SWIFT: Scattering With Identical Faddeev Three-body solver")
    println("    Modern Web UI using Blink.jl")
    println("="^70)
    println()

    # Create window
    global win = Window()
    title(win, "SWIFT Nuclear Calculator")
    size(win, 1920, 1080)

    # Load HTML
    body!(win, HTML_CONTENT)

    # Load Plotly.js by reading and injecting it
    println("Loading Plotly.js...")
    plotly_js = read("general_modules/plotly-2.33.0.min.js", String)
    Blink.js(win, Blink.JSString(plotly_js))
    sleep(0.5)  # Give it time to load
    println("✓ Plotly.js injected successfully!")

    # Register message handler
    handle(win, "run_calculation") do params
        # Start calculation in background, return immediately
        @async begin
            try
                run_calculation_web(params)
            catch e
                error_msg = sprint(showerror, e, catch_backtrace())
                println("Error in calculation: ", error_msg)
                # Send error to UI
                try
                    full_error = "<br><br>❌ ERROR:<br>" * error_msg
                    @js_ win document.getElementById("status").textContent = "❌ Calculation error"
                    @js_ win document.getElementById("status").className = "status error"
                    @js_ win document.getElementById("output").innerHTML += $full_error
                    @js_ win document.getElementById("run-btn").disabled = false
                catch js_err
                    println("Could not send error to UI: ", js_err)
                end
            end
        end
        return nothing  # Return immediately to unblock JavaScript
    end

    println("\n✓ Web UI launched successfully!")
    println("  Adjust parameters and click 'Run Calculation' to start.")
    println("  Close the window to exit.")
    println()

    # Keep window open
    try
        while active(win)
            sleep(0.1)
        end
    catch e
        if isa(e, InterruptException)
            println("\n\nShutting down...")
        else
            rethrow(e)
        end
    end
end

# Run the GUI
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
