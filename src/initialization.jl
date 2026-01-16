# Initialization utilities for FOMPrototypes

########################
# Initialization Block #
########################

function initialise_misc(backend::Symbol = :plotlyjs)
    # Set Plots backend.
    # For interactive plots: plotlyjs()
    # For faster plotting: gr()
    RES = 500
    if backend == :plotlyjs
        plotlyjs(dpi=RES)
    elseif backend == :plotly
        plotly(dpi=RES)
    elseif backend == :pyplot
        pyplot(dpi=RES)
    elseif backend == :gr
        gr(dpi=RES)
    else
        error("Invalid backend specified. Use :plotlyjs, :pyplot, or :gr.")
    end

    # Determine newline character based on backend.
    local newline_char = Plots.backend_name() in [:gr, :pythonplot] ? "\n" : "<br>"

    # Set default plot size (in pixels)
    # default(size=(2000, 450)) # for desktop
    default(size=(800, 600)) # for laptop

    return newline_char
end

##################################
# Problem Selection              #
##################################

function choose_problem(problem_option::Symbol)
    # Choose problem option. Valid options: :LASSO, :HUBER, :MAROS, :GISELSSON

    if problem_option === :LASSO
        problem_set = "sslsq"
        problem_name = "NYPA_Maragal_5_lasso"; # large, challenging
        # problem_name = "HB_abb313_lasso"  # (m, n) = (665, 665)
        # problem_name = "HB_ash219_lasso" # (m, n) = (389, 389)
    elseif problem_option === :HUBER
        problem_set = "sslsq"
        problem_name = "HB_ash958_huber"  # (m, n) = (3419, 3099)
    elseif problem_option === :MAROS
        problem_set = "maros"
        # problem_name = "DUAL3"; # large
        problem_name = "QSCSD8"   # not as large, (m, n) = (3147, 2750)
        # Other MAROS options commented out...
    elseif problem_option === :GISELSSON
        problem_set = "giselsson"
        problem_name = "giselsson_problem"
    else
        error("Invalid problem option")
    end

    return problem_option, problem_set, problem_name
end
