import os
import sys
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add parent directory to path to import from src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.benchmark_runner import (
    GreedySampling,
    MCMCSampling,
    TemperatureSampling,
    BeamSearchSampling,
)
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Reasoning with Sampling - Interactive Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .method-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stats {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🧠 Reasoning with Sampling</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare different sampling strategies side-by-side</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    st.subheader("API Settings")
    api_key = st.text_input(
        "X.AI API Key",
        value=os.getenv("XAI_API_KEY", ""),
        type="password",
        help="Get your API key from https://console.x.ai/"
    )
    
    model_name = st.selectbox(
        "Model",
        ["grok-2-1212", "grok-beta", "grok-2-latest"],
        index=0
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2048,
        value=512,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.divider()
    
    # Sampling Methods Selection
    st.subheader("Sampling Methods")
    
    use_greedy = st.checkbox("Greedy Decoding", value=True)
    use_mcmc = st.checkbox("MCMC Power Sampling", value=True)
    use_beam_search = st.checkbox("Beam Search", value=True)
    use_temperature = st.checkbox("Temperature Sampling", value=False)
    
    st.divider()
    st.markdown("""
    ### About
    This demo compares different sampling strategies from the paper 
    [**Reasoning with Sampling**](https://arxiv.org/abs/2510.14901).
    
    - **Greedy**: Deterministic (temp=0)
    - **MCMC**: Power sampling with Metropolis-Hastings
    - **Beam Search**: Best-first search across multiple candidates
    - **Temperature**: Standard sampling (optional)
    """)

# Main content area
st.header("💬 Prompt")
prompt = st.text_area(
    "Enter your prompt",
    height=150,
    placeholder="Example: Write a Python function to compute the Fibonacci sequence...",
    help="Enter the prompt you want to test with different sampling methods"
)

# Run button
run = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = {}

if run and prompt.strip():
    if not api_key:
        st.error("⚠️ Please enter your X.AI API key in the sidebar")
    else:
        # Initialize client
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        client.default_model = model_name
        
        # Build strategies list to determine layout
        strategy_configs = []
        if use_greedy:
            strategy_configs.append(("Greedy Decoding", "greedy", None))
        if use_mcmc:
            strategy_configs.append(("MCMC Power Sampling", "mcmc", None))
        if use_beam_search:
            strategy_configs.append(("Beam Search", "beam_search", None))
        if use_temperature:
            strategy_configs.append(("Temperature Sampling", "temperature", None))
        
        if not strategy_configs:
            st.warning("⚠️ Please select at least one sampling method")
        else:
            st.header("📊 Results")
            
            # Create columns for side-by-side comparison
            cols = st.columns(len(strategy_configs))
            
            # Create placeholders for each column and collect method-specific settings
            placeholders = []
            stats_containers = []
            strategies = []
            
            for idx, (col, (name, method_type, _)) in enumerate(zip(cols, strategy_configs)):
                with col:
                    st.subheader(name)
                    
                    # Method-specific settings in expander
                    with st.expander("⚙️ Settings", expanded=False):
                        if method_type == "mcmc":
                            alpha = st.slider(
                                "Alpha (α)",
                                min_value=1.0,
                                max_value=8.0,
                                value=4.0,
                                step=0.5,
                                help="Power for target distribution p^α",
                                key=f"alpha_{idx}"
                            )
                            
                            mcmc_steps = st.slider(
                                "MCMC Steps",
                                min_value=1,
                                max_value=20,
                                value=10,
                                step=1,
                                help="Number of MCMC refinement steps per block",
                                key=f"mcmc_steps_{idx}"
                            )
                            
                            block_size = st.slider(
                                "Block Size",
                                min_value=32,
                                max_value=512,
                                value=192,
                                step=32,
                                help="Block size for block-wise generation",
                                key=f"block_size_{idx}"
                            )
                            
                            proposal_temperature = st.slider(
                                "Proposal Temperature",
                                min_value=0.1,
                                max_value=2.0,
                                value=1.0,
                                step=0.1,
                                help="Temperature for MCMC proposal distribution",
                                key=f"proposal_temp_{idx}"
                            )
                            
                            debug_mcmc = st.checkbox("Debug MCMC", value=False, key=f"debug_{idx}")
                            
                            strategy = MCMCSampling(
                                alpha=alpha,
                                mcmc_steps=mcmc_steps,
                                block_size=block_size,
                                proposal_temperature=proposal_temperature,
                                debug=debug_mcmc
                            )
                            strategy_name = f"MCMC (α={alpha}, steps={mcmc_steps})"
                            
                        elif method_type == "beam_search":
                            num_beams = st.slider(
                                "Number of Beams",
                                min_value=2,
                                max_value=10,
                                value=4,
                                step=1,
                                help="Number of parallel sequences to generate",
                                key=f"num_beams_{idx}"
                            )
                            
                            beam_temperature = st.slider(
                                "Beam Temperature",
                                min_value=0.1,
                                max_value=2.0,
                                value=0.7,
                                step=0.1,
                                help="Temperature for beam search sampling",
                                key=f"beam_temp_{idx}"
                            )
                            
                            strategy = BeamSearchSampling(num_beams=num_beams, temperature=beam_temperature)
                            strategy_name = f"Beam Search (beams={num_beams})"
                            
                        elif method_type == "temperature":
                            temperature = st.slider(
                                "Temperature",
                                min_value=0.1,
                                max_value=2.0,
                                value=0.8,
                                step=0.1,
                                help="Higher = more random, Lower = more deterministic",
                                key=f"temperature_{idx}"
                            )
                            
                            strategy = TemperatureSampling(temperature=temperature)
                            strategy_name = f"Temperature (T={temperature})"
                            
                        else:  # greedy
                            st.info("No configuration needed - uses temperature=0")
                            strategy = GreedySampling()
                            strategy_name = "Greedy Decoding"
                    
                    strategies.append((strategy_name, strategy))
                    
                    placeholders.append({
                        'header': st.empty(),
                        'spinner': st.empty(),
                        'response': st.empty(),
                        'error': st.empty()
                    })
            
            # Create a separate row for stats (will be aligned)
            st.markdown("---")
            stats_cols = st.columns(len(strategy_configs))
            for col in stats_cols:
                with col:
                    stats_containers.append(st.empty())
            
            # Function to run a single strategy
            def run_strategy(name, strategy):
                try:
                    start_time = time.time()
                    completion, prompt_tokens, completion_tokens = strategy.generate(
                        client,
                        prompt,
                        max_tokens=max_tokens
                    )
                    elapsed_time = time.time() - start_time
                    
                    result = {
                        'completion': completion,
                        'tokens': prompt_tokens + completion_tokens,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'time': elapsed_time,
                        'error': None
                    }
                    
                    # Get MCMC-specific stats
                    if isinstance(strategy, MCMCSampling):
                        result['acceptance_ratio'] = strategy.get_acceptance_ratio()
                    
                    return name, result
                    
                except Exception as e:
                    return name, {'error': str(e)}
            
            # Run all strategies in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
                # Submit all tasks
                future_to_strategy = {
                    executor.submit(run_strategy, name, strategy): (idx, name)
                    for idx, (name, strategy) in enumerate(strategies)
                }
                
                # Show initial state
                for idx, (name, _) in enumerate(strategies):
                    placeholders[idx]['header'].subheader(name)
                    placeholders[idx]['spinner'].info("⏳ Generating...")
                
                # Process results as they complete
                for future in as_completed(future_to_strategy):
                    idx, name = future_to_strategy[future]
                    strategy_name, result = future.result()
                    
                    # Clear spinner
                    placeholders[idx]['spinner'].empty()
                    
                    if result.get('error'):
                        # Show error
                        placeholders[idx]['error'].error(f"❌ Error: {result['error']}")
                    else:
                        # Method description
                        descriptions = {
                            'Greedy Decoding': 'Selects the most probable token at each step, providing deterministic and focused outputs. Best for tasks requiring consistency.',
                            'MCMC': 'Uses Metropolis-Hastings to sample from p^α distribution. Explores high-probability regions through iterative refinement with accept/reject steps.',
                            'Beam Search': 'Maintains multiple parallel hypotheses and selects the best based on cumulative probability scores. Balances exploration and exploitation.',
                            'Temperature': 'Introduces controlled randomness by scaling logits. Higher temperature increases diversity; lower increases focus.'
                        }
                        
                        # Find matching description
                        method_desc = None
                        for key in descriptions:
                            if key in strategy_name:
                                method_desc = descriptions[key]
                                break
                        
                        if method_desc:
                            placeholders[idx]['header'].markdown(
                                f"_{method_desc}_",
                                unsafe_allow_html=False
                            )
                        
                        # Display result
                        response_html = f"""
                        <div class="method-card" style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1rem;">
                            {result["completion"]}
                        </div>
                        """
                        placeholders[idx]['response'].markdown(response_html, unsafe_allow_html=True)
                        
                        # Stats in aligned row below (no background, clean table-like layout)
                        stats_parts = [
                            '<div style="padding: 0.5rem 0;">',
                            f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                            f'<strong>📝 Tokens:</strong> {result["tokens"]}',
                            f'<div style="color: #666; font-size: 0.85rem; margin-left: 1.5rem;">',
                            f'prompt: {result["prompt_tokens"]}, completion: {result["completion_tokens"]}',
                            f'</div>',
                            f'</div>',
                            f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                            f'<strong>⏱️ Time:</strong> {result["time"]:.2f}s',
                            f'</div>',
                            f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                            f'<strong>📏 Length:</strong> {len(result["completion"])} chars',
                            f'</div>',
                        ]
                        
                        # MCMC-specific stats
                        if 'acceptance_ratio' in result:
                            stats_parts.extend([
                                f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                                f'<strong>✅ Acceptance:</strong> {result["acceptance_ratio"]:.1%}',
                                f'</div>',
                            ])
                        
                        stats_parts.append('</div>')
                        stats_html = ''.join(stats_parts)
                        
                        stats_containers[idx].markdown(stats_html, unsafe_allow_html=True)
                        
                        results[strategy_name] = result
            
            # Store results in session state
            st.session_state.results = results

elif run and not prompt.strip():
    st.warning("⚠️ Please enter a prompt")

# Show previous results if available
if not run and st.session_state.results:
    st.info("💡 Previous results are still available. Modify settings and run again to compare.")
