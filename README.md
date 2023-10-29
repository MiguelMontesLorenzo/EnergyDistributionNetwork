# EnergyDistributionNetwork

---

## 1. Introduction to the Problem

Modern electrical grids are complex systems that need to effectively distribute energy to meet the demand of their consumers. Optimizing the distribution of energy is not only crucial for ensuring the reliability of the grid but also for minimizing losses and costs.

In this project, we've tackled the challenge of optimizing the energy distribution in an electrical grid. To make the problem tractable, several simplifications have been made. Here are some of the main assumptions and their physical implications:

---

## 2. Simplifications and Implications

- **Linearization of Power Flow Equations:** While the true nature of power flows in a network is governed by nonlinear equations, we've linearized them for the sake of simplicity. This means our model may not perfectly capture all the intricacies of real-world power flows, but it offers a good approximation for most scenarios.
  
- **Constant Voltage:** We assume that the voltage at each node remains constant. This simplifies our calculations but omits the potential voltage variations that can occur in the real world, which can affect power quality and stability.
  
- **Single Time Frame:** Our optimization is based on a single time frame, ignoring the potential fluctuations in demand and supply that can occur throughout the day. This means the model might not capture the dynamic nature of the grid but provides an average optimal solution.

---

## 2. Problem Modelization and Formulation

### Sets

- **Node (i,j)**: Nodes in the network { $P1,P2,\ldots, T1,T2,\ldots, C1,C2,\ldots, k1,k2,\ldots$ }
- **PrTy (p)**: Production type { thermal, hydraulic, solar, wind, none} 
- **Time (t)**: Time { $0, \ldots, 23$ }

### Parameters

- $DEM_{c,t}$: Electrical demand of sector c at time t [MWh]
- $MAXPROD_{i}$: Maximum production capacity of node i [MWh]
- $MINPROD_{i}$: Minimum production of node i [MWh]
- $THRESHOLD_{i}$: Production required if node i is active [MWh]
- $SOLAR_{i,t}$: Solar production of node i at time t [MWh]
- $WIND_{i,t}$: Wind power production of node i at time t [MWh]
- $UNITARYCOST_{p}$: Unitary production cost of unit type p [€/MWh]
- $FXCOST_{p}$: Fixed cost of keeping a node type p working [€]
- $TRNONCOST_{p}$: Cost of turning on node type p [€]
- $TRNOFFCOST_{p}$: Cost of turning off unit type p [€]
- $W{i,p}$: Unit type [2D {0,1}]
- $CONN_{K,c_t}$: Exist connection between i and j {0,1}
- $L_{i,j}$: Length of cable ij [km]
- $V_{i,j}$: Voltage of cable ij [kV]
- $R_{i,j}$: Cable ij resistance [Ω/km] ~15
- $H_{i}$: Height of hydraulic press water at time t [m]
- $PH_{i}$: Relationship between height of hydraulic press and maximum production [m] —> [MWh]
- $F_{i}$: Minimum flowing water [m³]

### Variables

- $p_{i,t}$: Production of node i at time t [MWh]
- $a_{i,t}$: Node i active (producing) at time t {0,1}
- $on_{i,t}$: Time when node i production is turned on {0,1}
- $off_{i,t}$: Time when node i production is turned off {0,1}
- $e_{i,j,t}$: Energy transferred from i to j at time t [MWh]
- $loss_{i,j,t}$: Energy loss in cable ij [%]
- $intensity_{i,j,t}$: Current intensity on cable ij [A] (conversion needed)
- $dailyHyd_{u}$: Energy to be produced in each hydraulic press the whole day [MWh]

### Objective Function

$\text{minimize} \sum_{u,t} W_{u,p} \left( p_{u,t} \times UNITARYCOST_{p} + a_{u,t} \times FXCOST_{p} + on_{u,t} \times TRNONCOST_{p} + off_{u,t} \times TRNOFFCOST_{p} \right)$

### Constraints

1. **Total demand covered (Kirchoff 1):**
$\sum_{i} e_{i,j,t} \leq p_{i,t} - DEM_{i,t}$

2. **Production must cover losses (Kirchoff 2):**
$\sum_{i} e_{i,j,t} + loss_{i,j,t} \leq p_{i,t}$

3. **Power in line = Powerflow + loss:**
$e_{i,j,t} + loss_{i,j,t} = V_{i,j} \times intensity_{i,j,t}$

4. **PowerFlow sign depends on direction:**
$e_{i,j,t} = (- e_{j,i,t})$
$loss_{i,j,t} = (- loss_{j,i,t})$

5. **Loss definition:**
$loss_{i,j,t} = R_{i,j} \times L_{i,j} \times intensity_{i,j,t}$

6. **Total PowerFlow in function of VI:**
$e_{i,j,t} = (V_{i,j} - R_{i,j} \times L_{i,j}) \times intensity_{i,j,t}$

7. **PowerFlow limits:**
$- MAXPOWERFLOW_{i,j} < e_{i,j,t} < MAXPOWERFLOW_{i,j}$

8. **No PowerFlow between non-connected nodes:** (if $CONN_{i,j} = 0$)
$R_{i,j} = 0, V_{i,j} = 0, loss_{i,j,t} = 0, e_{i,j,t} = 0$

9. **Positive production:**
$p_{i,t} \geq 0$

10. **Define maximum productions:**
$p_{i,t} = \sum W_{i,p} \times MAXPROD_{p}$

Sure, continuing with the constraints:

11. **Turn on / Turn off (only if node type is $p = \text{'thermal'}$):**
   - Maximum production if active:
     $     p_{i,t} \leq MAXPROD_p \times a_{i,t}
     \]
   - Minimum production if active:
     $     p_{i,t} \geq THRESHOLD_p \times a_{i,t}
     \]
   - Turning off:
     $     a_{i,t-1} \geq a_{i,t} + \varepsilon \Leftrightarrow \text{off} = 1
     \]
     Which can be linearized as:
     $     a_{i,t-1} \geq a_{i,t} + \varepsilon + M \times \text{off}_{i,t}
     \]
     and
     $     \text{off} = 1 \Rightarrow a_{i,t-1} \geq a_{i,t} + \varepsilon + m \times (1-\text{off}_{i,t})
     \]
   - Turning on:
     $     a_{i,t} \geq a_{i,t-1} + \varepsilon \Leftrightarrow \text{on} = 1
     \]
     Which can be linearized as:
     $     a_{i,t} \leq a_{i,t-1} + \varepsilon + M \times \text{on}_{i,t}
     \]
     and
     $     \text{on} = 1 \Rightarrow a_{i,t} \geq a_{i,t-1} + \varepsilon + m \times (1-\text{on}_{i,t})
     \]

12. **Hydraulic production matches daily stipulated:**
   - Total hydraulic production:
     $     \sum_{t} W_{i,p} \times p_{i,t} = \text{dailyHyd}_{i} \times W_{i,p}
     \]
   - Daily stipulated production:
     $     \text{dailyHyd}_{i} = PH_{i} \times H_{i}
     \]

13. **Solar and wind productions are fixed:**
   - Solar production:
     $     p_{i,t} \times W_{i,p} = SOLAR_{i,t} \times W_{i,p}
     \]
   - Wind production:
     $     p_{i,t} \times W_{i,p} = WIND_{i,t} \times W_{i,p}
     \]



### Scripts of the Project

1. **main.py**
    - **Funciones/Métodos**:
        - `load_and_prepare_data`: Carga y prepara los datos necesarios para el modelo.
        - `load_and_prepare_dictionaries`: Convierte los datos cargados en diccionarios para su fácil manejo.
        - `generate_graph`: Genera un gráfico visual de la red eléctrica y su estado actual.
        - `create_video`: Crea un video visualizando la optimización a lo largo del tiempo.
    - **Tareas**:
        - Preparar datos.
        - Instanciar y optimizar el modelo.
        - Mostrar los resultados optimizados.
        - Crear un video de la evolución de la optimización.

2. **ego.py**
    - **Funciones/Métodos**:
        - [...] (Por proporcionar...)
    - **Tareas**:
        - [...] (Por proporcionar...)

