# this file is used to run solver code directly from a terminal command
using Revise
import FOMPrototypes

if abspath(PROGRAM_FILE) == @__FILE__
    config = FOMPrototypes.parse_command_line()
    println()

    println("first run:")
    FOMPrototypes.main(config)

    println("second run:")
    FOMPrototypes.main(config)

    println("third run:")
    config["problem-name"] = "NYPA_Maragal_5_lasso"
    FOMPrototypes.main(config)
end