import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for nicer defaults

def show_prediction_bar(probabilities: dict, class_colors: dict, title="Prediction Confidence"):
    """
    Displays a horizontal stacked bar chart of prediction probabilities.
    Includes percentage labels on each bar segment.
    """
    # Sort probabilities by class name to ensure consistent order if needed,
    # or by probability value to highlight the highest ones.
    # For a stacked bar, sorting by class name usually makes sense.
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[0]) # Sort by class name

    # Set up the figure for the bar chart.
    # Adjust figsize for compactness. (width, height) in inches.
    # A width of 3 to 4 inches often works well in Streamlit columns.
    fig, ax = plt.subplots(figsize=(3.5, 0.6)) # Adjusted width for compactness

    left_val = 0
    for cls, pct in sorted_probs:
        color = class_colors.get(cls, '#CCCCCC') # Default to a light gray if color not found
        ax.barh(0, pct, left=left_val, color=color, edgecolor='white', linewidth=0.5)

        # Add percentage label on the bar segment if it's large enough
        if pct > 5: # Only label if segment is at least 5% to avoid clutter
            ax.text(left_val + pct / 2, 0, f'{pct:.1f}%',
                    ha='center', va='center', color='black', fontsize=7, weight='bold')
        left_val += pct

    # --- Aesthetic Improvements ---
    ax.set_xlim(0, 100) # Ensure the bar spans 0 to 100%
    ax.set_xticks([]) # Remove x-axis ticks
    ax.set_yticks([]) # Remove y-axis ticks

    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Make the background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Set title. Use a slightly smaller font size for compactness.
    ax.set_title(title, fontsize=8, pad=5) # Reduced fontsize and padding

    # Use tight_layout to minimize whitespace around the plot
    fig.tight_layout(pad=0.5)

    st.pyplot(fig) # Display the matplotlib figure in Streamlit
    plt.close(fig) # Close the figure to free up memory

def show_model_result(model_name, predicted_class, confidence, probabilities, class_colors):
    """
    Displays the prediction results for a single model, including the class, confidence,
    and a styled probability bar chart.
    """
    # Using markdown with bold text for predicted class and confidence
    st.markdown(f"**{model_name}**") # Make model name bold
    st.markdown(f"Predicted: **{predicted_class}** (Confidence: **{confidence:.2%}**)")

    # Call the improved bar chart function
    # Pass the full probabilities dict directly to allow sorting inside the function
    show_prediction_bar(probabilities, class_colors, title="Probabilities")