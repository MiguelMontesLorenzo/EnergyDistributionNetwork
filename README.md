# Energy Distribution Network Optimization

---

## 1. Introduction to the Problem

Electric grids are intricate networks responsible for transmitting electricity from production facilities to consumers. These networks comprise various nodes, including power production units, transmission lines, and consumption sectors. Optimizing the energy distribution within such a network is a critical task, ensuring that electricity is delivered efficiently, economically, and reliably.

In this project, we tackle the optimization problem of energy distribution within an electrical grid. The main goal is to minimize the total operational cost of the network while satisfying energy demands, maintaining grid stability, and respecting the physical constraints of the infrastructure.

---

## 2. Simplifications and Implications

- **Linearization of Power Flow Equations:** While the true nature of power flows in a network is governed by nonlinear equations, we've linearized them for the sake of simplicity. As a general rule, distribution optimisation problems using a network require equations associated with the balance of incoming and outgoing quantities at each node. In the case of electrical energy distribution problems, the magnitude with which we work is the PowerFlow $S_{i,j}$ in the lines connecting each node, which is defined by the product of the potential at the node $V_{i,j}$ and the intensity of the line $I_{i,j} = V_{i,j} / R_{ij}$. Thus, the balance equations at the nodes (Kirchoff's Laws) have a non-linear behaviour.

   $S_{i,j} = V_{i} I_{i,j} = V^2_{i,j} / R_{i,j}$
  
   $\text{Kirchoff Law: } \quad \sum_{j} S_{i,j} = S^p_i - S^c_{i}$

   For this reason, in this project the potential drop between nodes $V_{i,j}$ will be taken as a constant of the line, and the PowerFlow will be defined as the product of the current at the node and the potential drop.

   $S_{i,j} = V_{i,j} I_{i,j} \\ \\ \alpha \\ \\ I_{i,j}$

   ###### *Note: As a consequence of the discretisation of the problem, hourly time intervals are used instead of continuous time. For consistency in the following the power* $e_{i,j,t}$ *will be expressed in units of energy [kwh] (power per hour).*

   This simplification implies that in this project there is neither the concept of potential at a node nor the concept of Ohm's law in a voltage line. This avoids the need to maintain a decreasing potential from production nodes to consumption nodes in order to generate currents that transport energy, but rather, in some way, we are directly manipulating the currents. In reality, there is a reason why currents alone cannot be increased at the consumption nodes: a very high current accompanied by a very low voltage is a current incapable of crossing large impedances, and therefore incapable of feeding energy-intensive processes.
  
- **Hidraulic Power:** The power generated by a dam is defined by the product of its efficiency $eta$, the density of the water $rho$, the intensity of the gravitational field at the earth's surface $g$, its flow $Q$ and its height $H$. Where the flow rate $Q$ and head $H$ are non-fixed variables whose interaction introduces non-linear behaviour into the equation.

   $P = \eta \times \rho \times g \times Q \times H$

  In this project, the power generated by the hydraulic production centres will be modelled as a variable proportional to the height of the dam.

   $P \\ \\ \alpha \\ \\ H$
  

---

## 3. Problem Modelization and Formulation

### Sets

- $Nodes \\ (i,j)$: Nodes in the network { $P1,P2,\ldots, T1,T2,\ldots, C1,C2$ }
- $PrTy \\ (p)$: Production type { $\text{thermal}, \text{hydraulic}, \text{solar}, \text{eolic}, \text{none}$ } 
- $Time \\ (t)$: Time { $1, \ldots, 24$ }

### Parameters

- $DEM_{i,t}$: Electrical demand of node $i$ at time $t \quad [\text{kwh}]$
- $MAXPROD_{i}$: Maximum production capacity of node $i \quad [\text{kwh}]$
- $MINPROD_{i}$: Minimum production of node $i \quad [\text{kwh}]$
- $THRESHOLD_{i}$: Production required if node $i$ is active $\quad [\text{kwh}]$
- $SOLAR_{i,t}$: Solar production of node $i$ at time $t \quad [\text{kwh}]$
- $WIND_{i,t}$: Wind power production of node $i$ at time $t \quad [\text{kwh}]$
- $UNITARYCOST_{p}$: Unitary production cost of unit type $p \quad [€/\text{kwh}]$
- $FXCOST_{p}$: Fixed cost of keeping a node type $p$ working $\quad [€]$
- $TRNONCOST_{p}$: Cost of turning on node type $p \quad [€]$
- $TRNOFFCOST_{p}$: Cost of turning off unit type $p \quad [€]$
- $W{i,p}$: Unit type [2D { $0,1$ } ]
- $CONN_{i,j}$: Exist connection between $i$ and $j \quad$ { $0,1$ }
- $L_{i,j}$: Length of cable $ij \quad [km]$
- $V_{i,j}$: Voltage of cable $ij \quad [kV]$
- $R_{i,j}$: Cable $ij$ resistance $\quad [\Omega/km]$
- $H_{i}$: Height of hydraulic press water at time $t \quad [m]$
- $PH_{i}$: Relationship between height of hydraulic press and maximum production $\quad [\text{kwh}/m]$
- $F_{i}$: Minimum flowing water $\quad [m^3]$

### Variables

- $p_{i,t}$: Production of node $i$ at time $t \quad [\text{kwh}]$
- $a_{i,t}$: Node $i$ active (producing) at time $t \quad$ { $0,1$ }
- $on_{i,t}$: Time when node $i$ production is turned on $\quad$ { $0,1$ }
- $off_{i,t}$: Time when node $i$ production is turned off $\quad$ { $0,1$ }
- $e_{i,j,t}$: Energy transferred from $i$ to $j$ at time $t \quad [\text{kwh}]$
- $loss_{i,j,t}$: Energy loss in cable $ij \quad [\text{kwh}]$
- $intensity_{i,j,t}$: Current intensity on cable $ij \quad [A]$ 
- $dailyHyd_{i}$: Energy to be produced in each hydraulic press the whole day $\quad [\text{kwh}]$

### Objective Function
- $\text{minimize} \sum_{i,t} W_{i,p} \left( p_{i,t} \times UNITARYCOST_{p} + a_{i,t} \times FXCOST_{p} + on_{i,t} \times TRNONCOST_{p} + off_{i,t} \times TRNOFFCOST_{p} \right)$

### Constraints

1. **Total demand covered (Kirchoff 1):**

     - $\sum_{j} e_{i,j,t} \leq p_{j,t} - DEM_{j,t} \quad \forall i,t$

2. **Production must cover losses (Kirchoff 2):**

     - $\sum_{j} e_{i,j,t} + loss_{i,j,t} \leq p_{j,t} \quad \forall i,t$

3. **Power in line = Powerflow + loss:**

     - $e_{i,j,t} + loss_{i,j,t} = V_{i,j} \times intensity_{i,j,t} \quad \forall i,j,t$

4. **PowerFlow sign depends on direction:**

     - $e_{i,j,t} = (- e_{j,i,t}) \quad \forall i,j,t$
     - $loss_{i,j,t} = (- loss_{j,i,t}) \quad \forall i,j,t$

5. **Define Loss as proportional to **$I$**:**

     - $loss_{i,j,t} = R_{i,j} \times L_{i,j} \times intensity_{i,j,t} \quad \forall i,j,t$

6. **PowerFlow limits:**

     - $- MAXPOWERFLOW_{i,j} < e_{i,j,t} < MAXPOWERFLOW_{i,j} \quad \forall i,j,t$

7. **No PowerFlow between non-connected nodes:** (if $CONN_{i,j} = 0$)

      - $R_{i,j} = 0, \\ V_{i,j} = 0, \\ loss_{i,j,t} = 0, \\ e_{i,j,t} = 0 \quad \forall i,j,t$

8. **Positive production:**

     - $p_{i,t} \geq 0 \quad \forall i,t$

9. **Define maximum productions:**

     - $p_{i,t} = \sum W_{i,p} \times MAXPROD_{p} \quad \forall i,t$

10. **Turn on / Turn off (only if node type is $p = \text{thermal}$):**
   - Maximum production if active:

      - $p_{i,t} \leq MAXPROD_p \times a_{i,t} \quad \forall i,t$
     
   - Minimum production if active:
    
      - $p_{i,t} \geq THRESHOLD_p \times a_{i,t} \quad \forall i,t$
     
   - Turning off:
     
      - $a_{i,t-1} \geq a_{i,t} + \varepsilon \Leftrightarrow \text{off} = 1 \quad \forall i,t$
     
      - Which can be linearized as:
     
         - $a_{i,t-1} \geq a_{i,t} + \varepsilon + M \times \text{off}_{i,t}$
        
         - $a_{i,t-1} \geq a_{i,t} + \varepsilon + m \times (1-\text{off}_{i,t})$
     
   - Turning on:
     
      - $a_{i,t} \geq a_{i,t-1} + \varepsilon \Leftrightarrow \text{on} = 1 \quad \forall i,t$
     
      - Which can be linearized as:
     
         - $a_{i,t} \leq a_{i,t-1} + \varepsilon + M \times \text{on}_{i,t}$
     
         - $a_{i,t} \geq a_{i,t-1} + \varepsilon + m \times (1-\text{on}_{i,t})$

11. **Hydraulic production matches daily stipulated:**
   - Total hydraulic production:

      - $\text{IF } p = hydraulic \text{:}$
      - $\sum_{t} p_{i,t} = dailyHyd_i \quad \forall i$
     
   - Daily stipulated production:
    
      - $dailyHyd_{i} = PH_{i} \times H_{i}$

12. **Solar and wind productions are fixed:**
    
   - Solar production:
      - $\text{IF } p = solar \text{:}$
      - $p_{i,t} \times W_{i,p} = SOLAR_{i,t} \times W_{i,p}$
     
   - Eolic production:
      - $\text{IF } p = eolic \text{:}$
      - $p_{i,t} \times W_{i,p} = WIND_{i,t} \times W_{i,p}$


---

## 4. Scripts of the Project

1. **main.py**
    - **Functions/Methods**:
        - `load_and_prepare_data`: Loads and prepares the necessary data for the model.
        - `load_and_prepare_dictionaries`: Converts the loaded data into dictionaries for easy handling.
        - `generate_graph`: Generates a visual graph of the electrical network and its current status.
        - `create_video`: Creates a video visualizing the optimization over time.
    - **Tasks**:
        - Prepare data.
        - Instantiate and optimize the model.
        - Display optimized results.
        - Create a video of the optimization's evolution.

2. **ego.py**
    - **Class**: `ElectricGridOptimization`
    - **Functions/Methods**:
        - `__init__`: Initializes the optimization problem with provided parameters and sets.
        - `define_sets`: Defines sets used in the optimization model.
        - `define_params`: Defines parameters used in the optimization model.
        - `define_variables`: Defines decision variables used in the optimization model.
        - `define_objective`: Defines the objective function of the optimization model.
        - `define_constraints`: Defines the constraints of the optimization model.
        - `define_solver`: Sets up the solver for optimization.
        - `define_solver_path`: Defines the path of the solver executable.
        - `optimize_problem`: Runs the optimization process and returns the results.
        - `show_results`: Prints the results of the optimization.
        - `display_constraints`: Displays the constraints used in the optimization model.
    - **Tasks**:
        - Model the optimization problem of an electric grid.
        - Define the structure of the optimization model (variables, constraints, objective).
        - Solve the optimization problem.
        - Display and interpret the results.


