from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

SAMPLE_TEXT = """

Formula 1: The Apex of Global Motorsport
Formula 1 represents the absolute pinnacle of open-wheel, single-seater international auto racing. Governing bodies, engineers, strategists, and world-class drivers collide in a multi-billion-dollar global circus that bridges high-stakes athletic endurance with cutting-edge aerospace engineering. Sanctioned by the Fédération Internationale de l'Automobile (FIA) since its inaugural championship season in 1950, Formula 1 operates as both a premier athletic discipline and a relentless technological testbed where a single millisecond can dictate the threshold between historical glory and devastating defeat.

Championship Structure and Weekend Anatomy
An F1 season takes place across a grueling global calendar spanning roughly twenty-four Grands Prix held on every continent except Antarctica. The competitive ecosystem is dual-faceted, featuring two distinct world titles awarded at the culmination of the season: the Drivers’ World Championship, awarded to the individual athlete who accumulates the highest total points, and the Constructors’ World Championship, awarded to the racing team that generates the highest combined point total across both of its entered cars.

A standard Grand Prix weekend unfolds over three consecutive days. Friday features two free practice sessions (FP1 and FP2), each lasting one hour. These sessions allow teams to validate aerodynamic upgrades, fine-tune suspension setups, test tire degradation rates on heavy fuel loads, and allow drivers to dial in their racing lines. Saturday begins with a final one-hour practice session (FP3) serving as a rehearsal for Qualifying, which takes place later that afternoon.

Qualifying determines the starting grid for Sunday's main event through a high-intensity, knockout-style format divided into three distinct segments:

Q1 (18 Minutes): All twenty cars on the grid hit the track. At the conclusion of the session, the five slowest drivers are eliminated and locked into grid positions 16 through 20.

Q2 (15 Minutes): The remaining fifteen drivers set their times, with the lowest five once again eliminated to secure grid positions 11 through 15.

Q3 (12 Minutes): A high-octane shootout among the top ten drivers to claim pole position—the coveted first spot on Sunday’s starting grid.

On select weekends throughout the season, Formula 1 utilizes a Sprint weekend format designed to increase competitive action. Under this alternative layout, Friday features a single practice session followed immediately by Sprint Qualifying (the "Sprint Shootout"). Saturday hosts a condensed, 100-kilometer Sprint Race that awards world championship points to the top eight finishers, followed by main Qualifying later in the day. Sunday’s Grand Prix remains the weekend's flagship event, covering approximately 305 kilometers (with the exception of Monaco) over a maximum two-hour window.

Points are awarded to the top ten finishers of the main Grand Prix according to a sliding scale: 25 points for first place, followed by 18, 15, 12, 10, 8, 6, 4, 2, and 1 point for tenth place. An additional single championship point is awarded to the driver who records the single fastest lap during the race, provided they finish within the top ten.

Engineering, Power Units, and Aerodynamics
Modern Formula 1 cars are complex technological marvels capable of generating lateral cornering forces exceeding 5G and braking from speeds over 300 km/h to zero in under three seconds. At the heart of these machines sits the hybrid Power Unit (PU), widely considered the most thermally efficient internal combustion system on Earth, achieving thermal efficiency levels well above 50 percent.

The current power unit framework consists of a 1.6-liter turbocharged 90-degree V6 internal combustion engine restricted to 15,000 RPM, working in synchronization with an Energy Recovery System (ERS). The ERS harvests waste thermal and kinetic energy using two electrical motor generator units:

MGU-K (Motor Generator Unit - Kinetic): Attached directly to the crankshaft, this unit recovers kinetic energy created under heavy braking, storing it as electrical energy in the onboard battery pack and deploying up to 120 kilowatts (roughly 160 horsepower) back into the drivetrain during acceleration.

MGU-H (Motor Generator Unit - Heat): Connected to the turbocharger, this device converts high-temperature waste energy from exhaust gases into electricity, which can be sent straight to the MGU-K or stored to spin up the compressor instantly, eliminating turbo lag.

Aerodynamics dominate modern F1 design, with teams relying on sophisticated computational fluid dynamics (CFD) and scale-model wind tunnels to maximize downforce—the invisible downward force pushed by airflow that presses the tires into the track surface—while minimizing drag. Modern regulations rely heavily on "ground effect" aerodynamics. Engineers shape the underside of the floor into sculpted Venturi tunnels. As air rushes beneath the car, it accelerates through these constrictions, causing a massive drop in pressure that effectively vacuums the chassis to the track, allowing for astonishing cornering speeds in sweeping curves.

To facilitate overtaking, cars feature a rear-wing Drag Reduction System (DRS). When a pursuing car is within one second of the car ahead at a designated detection point on track, the driver can hydraulically lift a flap in the center of the rear wing on specific straightaways. This sheds aerodynamic drag, yielding an extra 10 to 12 km/h in top speed until the driver applies the brakes.

Strategy, Pit Stops, and Tire Management
While performance on track is dictated by drivers, victories are frequently orchestrated on the pit wall through tire management and race strategy. Italian manufacturer Pirelli serves as the official single tire supplier, bringing three slick dry-weather compounds to each event chosen from a spectrum ranging from C1 (hardest) to C5 (softest):

Soft Compound (Red Sidewall): Maximum mechanical grip and ultimate speed, but suffers from rapid thermal degradation and graining.

Medium Compound (Yellow Sidewall): A balanced compound offering a middle ground between outright speed and stint durability.

Hard Compound (White Sidewall): Lower peak grip levels, but capable of surviving long stints with minimal performance decay.

Intermediate (Green) and Full Wet (Blue): Treaded tires utilized during rainfall capable of displacing dozens of liters of standing water per second.

Race regulations mandate that every driver must use at least two different dry tire compounds during a dry race, forcing a minimum of one pit stop. A modern F1 pit stop is a masterclass in human choreography: a crew of roughly twenty mechanics changes all four wheels in under two and a half seconds, with world-record pit stops occasionally dipping under 1.8 seconds.

Strategic depth revolves around timing these stops. A trailing driver might execute an undercut, pitting earlier than the leader for fresh rubber to set rapid out-laps; when the leader eventually pits a lap or two later, the chasing driver clears them on track. Conversely, the overcut involves staying out on older tires while a competitor struggles with warm-up or traffic on new tires, building a gap sufficient to pit and re-emerge ahead.

From the financial guardrails of the FIA Cost Cap to the high-G physical punishment inflicted on drivers' neck muscles, Formula 1 stands as an incredible synthesis of sport, science, and human endurance.
 """

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"\nFirst chunk:\n{chunks[0][:200]}\n")

if __name__ == "__main__":
    print("=== Recursive Splitter Demo ===")
    recursive_splitter()
