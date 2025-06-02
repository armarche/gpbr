import matplotlib.pyplot as plt
import numpy as np

def plot_radial_boundaries_2d(exact_rfunc, approx_rfunc, outter_rfunc, n_theta=256):
    """
    Plot exact and approximate curves with enhanced colors and styling.
    
    Parameters:
    -----------
    exact_rfunc : function
        Function for the exact solution radius
    approx_rfunc : function
        Function for the approximate solution radius
    outter_rfun : function
        Function for the outer boundary radius
    n_theta : int
        Number of points for plotting (resolution)
    title : str
        Main title for the plot
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the plot
    """
    # Create figure
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    thetas = np.linspace(0, 2*np.pi, n_theta)

    # Calculate radius values
    exact_rvals = exact_rfunc(thetas)
    approx_rvals = approx_rfunc(thetas)
    outter_rvals = outter_rfunc(thetas)

    # Convert to cartesian coordinates
    xx = np.cos(thetas)
    yy = np.sin(thetas)

    exact_x = exact_rvals * xx
    exact_y = exact_rvals * yy

    approx_x = approx_rvals * xx
    approx_y = approx_rvals * yy

    outter_x = outter_rvals * xx
    outter_y = outter_rvals * yy

    # Outer boundary - use a warm color (orange)
    ax1.plot(outter_x, outter_y, color='#FF8C00', linewidth=2.5, label='Outer Boundary',
             linestyle='-', alpha=0.8)
    
    # Exact solution - use a distinct blue
    ax1.plot(exact_x, exact_y, color='#1E88E5', linestyle='--', linewidth=3, 
             label='Exact Solution', alpha=0.9)

    # Approximation - use a vibrant purple
    ax1.plot(approx_x, approx_y, color='#8E24AA', linewidth=3, label='Approximation')
    
    # Figure styling
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlabel('X', fontsize=13)
    ax1.set_ylabel('Y', fontsize=13)
    
    # Add legend with custom ordering (to match the visual layering)
    handles, labels = ax1.get_legend_handles_labels()
    order = [2, 0, 1]  # Reorder to: Outer, Exact, Approx
    ax1.legend([handles[i] for i in order], [labels[i] for i in order], 
               loc='best', framealpha=0.9, fontsize=12)
    
    # Set axis limits based on the maximum extent
    max_radius = max(
        np.max(np.sqrt(exact_x**2 + exact_y**2)),
        np.max(np.sqrt(approx_x**2 + approx_y**2)),
        np.max(np.sqrt(outter_x**2 + outter_y**2)),
    )
    padding = max_radius * 0.15
    
    ax1.set_xlim([-max_radius-padding, max_radius+padding])
    ax1.set_ylim([-max_radius-padding, max_radius+padding])
    
    # Apply tight layout
    fig1.tight_layout()
    
    return fig1

def plot_evolution_stats(log, title="Evolutionary Algorithm Performance", figsize=(10, 6)):
    """
    Visualize the evolutionary process based on DEAP's MultiStatistics log,
    with each metric on a separate figure.
    
    Parameters:
    -----------
    log : DEAP logbook
        The logbook returned by DEAP's algorithms (eaSimple, eaMuPlusLambda, etc.)
    title : str, optional
        Base title for the plots
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    figs : tuple
        Tuple containing the three figure objects
    """
    import matplotlib.pyplot as plt
    
    # Extract generations
    gen = log.select("gen")
    
    # Extract statistics
    fit_mins = log.chapters["fitness"].select("min")
    fit_avgs = log.chapters["fitness"].select("avg")
    fit_maxs = log.chapters["fitness"].select("max")
    fit_stds = log.chapters["fitness"].select("std")
    
    size_avgs = log.chapters["size"].select("avg")
    size_mins = log.chapters["size"].select("min")
    size_maxs = log.chapters["size"].select("max")
    size_stds = log.chapters["size"].select("std")
    
    # Figure 1: Average Fitness Trend with std deviation only
    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(111)
    ax1.plot(gen, fit_avgs, 'b-', label='Average Fitness', linewidth=2.5)
    ax1.fill_between(gen, 
                     [a - b for a, b in zip(fit_avgs, fit_stds)],
                     [a + b for a, b in zip(fit_avgs, fit_stds)],
                     alpha=0.2, color='blue', label='Std Dev')
    
    # ax1.set_title(f'{title}: Fitness Evolution', fontsize=14)
    # ax1.set_title(f'Fitness Evolution', fontsize=14)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    fig1.tight_layout()
    
    # Figure 2: Solution Size Evolution
    fig2 = plt.figure(figsize=figsize)
    ax2 = fig2.add_subplot(111)
    ax2.plot(gen, size_avgs, 'm-', label='Average Size', linewidth=2.5)
    ax2.plot(gen, size_mins, 'c-', label='Min Size', linewidth=1.5)
    ax2.plot(gen, size_maxs, 'y-', label='Max Size', linewidth=1.5)
    
    # ax2.set_title(f'Solution Size Evolution', fontsize=14)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Tree Size', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    
    # Figure 3: Size vs. Fitness Correlation
    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(111)
    sc = ax3.scatter(size_avgs, fit_mins, c=gen, cmap='viridis', 
                    s=60, edgecolor='k', alpha=0.7)
    # sc = ax3.scatter(size_avgs, fit_avgs, c=gen, cmap='viridis', 
    #                 s=60, edgecolor='k', alpha=0.7)
    
    # Add colorbar
    cbar = fig3.colorbar(sc, ax=ax3)
    cbar.set_label('Generation')
    
    # Add arrows to show progression
    for i in range(1, len(gen)):
        ax3.annotate('',
                    xy=(size_avgs[i], fit_mins[i]),
                    xytext=(size_avgs[i-1], fit_mins[i-1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5))
    # # Add arrows to show progression
    # for i in range(1, len(gen)):
    #     ax3.annotate('',
    #                 xy=(size_avgs[i], fit_avgs[i]),
    #                 xytext=(size_avgs[i-1], fit_avgs[i-1]),
    #                 arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5))
    
    # ax3.set_title(f'Size-Fitness Correlation (best individual)', fontsize=14)
    ax3.set_xlabel('Average Solution Size', fontsize=12)
    ax3.set_ylabel('Average Fitness', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    fig3.tight_layout()
    
    return (fig1, fig2, fig3)