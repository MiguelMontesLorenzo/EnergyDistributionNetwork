import pyomo.environ as pyo
from typing import Dict, Any
import numpy as np

class ElectricGridOptimization:

    def __init__(self, sets: Dict[str, Any], params: Dict[str, Any]):
        """
        Initialize the optimization problem with the provided parameters and sets.
        """
        self.sets = sets
        self.params = params
        
        # Create model
        self.model = pyo.ConcreteModel()
        self.results = None
        self.obj_value = None

        # Control
        self.show_constraints = False

        self.production_type_indexes = {'normal': 1, 'hydraulic': 2, 'solar':3, 'eolic':4, 'none':5}
        self.turn_off_turn_on_rules_affected_types = ['normal']
        self.special_types = set(self.production_type_indexes.keys()) - set(self.turn_off_turn_on_rules_affected_types)

    def define_sets(self):
        """
        Define sets used in the model.
        """
        self.model.Nodes = pyo.Set(initialize=self.sets['Nodes'])
        self.model.PrTy = pyo.Set(initialize=self.sets['PrTy'])
        self.model.Time = pyo.Set(initialize=self.sets['Time'])

    def define_params(self):
        """
        Define parameters used in the model.
        """
        # Parameters with two indices
        self.model.DEM = pyo.Param(self.model.Nodes, self.model.Time, initialize=self.params['DEM'])
        self.model.SOLAR = pyo.Param(self.model.Nodes, self.model.Time, initialize=self.params['SOLAR'])
        self.model.WIND = pyo.Param(self.model.Nodes, self.model.Time, initialize=self.params['WIND'])
        
        # Parameters with single index
        self.model.MAXPROD = pyo.Param(self.model.Nodes, initialize=self.params['MAXPROD'])
        self.model.MINPROD = pyo.Param(self.model.Nodes, initialize=self.params['MINPROD'])
        self.model.THRESHOLD = pyo.Param(self.model.Nodes, initialize=self.params['THRESHOLD'])
        self.model.UNITARYCOST = pyo.Param(self.model.PrTy, initialize=self.params['UNITARYCOST'])
        self.model.FXCOST = pyo.Param(self.model.PrTy, initialize=self.params['FXCOST'])
        self.model.TRNONCOST = pyo.Param(self.model.PrTy, initialize=self.params['TRNONCOST'])
        self.model.TRNOFFCOST = pyo.Param(self.model.PrTy, initialize=self.params['TRNOFFCOST'])
        
        # Parameters with a node-node index
        self.model.CONN = pyo.Param(self.model.Nodes, self.model.Nodes, initialize=self.params['CONN'])
        self.model.L = pyo.Param(self.model.Nodes, self.model.Nodes, initialize=self.params['L'])
        self.model.V = pyo.Param(self.model.Nodes, self.model.Nodes, initialize=self.params['V'])
        self.model.R = pyo.Param(self.model.Nodes, self.model.Nodes, initialize=self.params['R'])
        self.model.MAXPOWERFLOW = pyo.Param(self.model.Nodes, self.model.Nodes, initialize=self.params['MAXPOWERFLOW'])
        
        # Parameters with single node index
        self.model.H = pyo.Param(self.model.Nodes, initialize=self.params['H'])
        self.model.PH = pyo.Param(self.model.Nodes, initialize=self.params['PH'])
        
        # 2D Parameters for unit type
        self.model.W = pyo.Param(self.model.Nodes, self.model.PrTy, initialize=self.params['W'])


    def define_variables(self):
        """
        Define variables used in the model.
        """
        # Production of unit u at time t 
        self.model.p = pyo.Var(self.model.Nodes, self.model.Time, within=pyo.NonNegativeReals)

        # production active in i + turned on + turned off
        self.model.a = pyo.Var(self.model.Nodes, self.model.Time, within=pyo.Binary)
        self.model.on = pyo.Var(self.model.Nodes, self.model.Time, within=pyo.Binary)
        self.model.off = pyo.Var(self.model.Nodes, self.model.Time, within=pyo.Binary)

        # Powerflow in cable ij at t + Energy loss in cable ij + Intensity in ij
        self.model.e = pyo.Var(self.model.Nodes, self.model.Nodes, self.model.Time, within=pyo.Reals)
        self.model.loss = pyo.Var(self.model.Nodes, self.model.Nodes, self.model.Time, within=pyo.Reals)
        self.model.intensity = pyo.Var(self.model.Nodes, self.model.Nodes, self.model.Time, within=pyo.Reals)

        # Energy to be produced in each hydraulic press the whole day
        self.model.dailyHyd = pyo.Var(self.model.Nodes, within=pyo.NonNegativeReals)


    def define_objective(self):
        """
        Define the objective function.
        """
        def objective_rule(model):
            return sum(model.W[i, p] * 
                    (
                        model.p[i, t] * model.UNITARYCOST[p] + 
                        model.a[i, t] * model.FXCOST[p] + 
                        model.on[i, t] * model.TRNONCOST[p] + 
                        model.off[i, t] * model.TRNOFFCOST[p]
                    )
                for i in model.Nodes for p in model.PrTy for t in model.Time)
        
        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Assuming that you will integrate this into the ElectricGridOptimization class.

    def define_constraints(self):
        """
        Define the constraints of the model.
        """
        
        # 1. Total demand covered (Kirchoff 1)
        def demand_covered_rule(model, i, t):
            # \sum_i e_ijt = p_it - DEM_it
            # return sum(model.e[i, j, t] for i in model.Nodes) == model.p[j, t] - model.DEM[j, t]
            # \sum_i e_ijt ≤ p_it - DEM_it   # Poweflow que sale debe ser menor a lo que se produce menos lo que se consume 
            return pyo.inequality(sum(model.e[i,j,t] for j in model.Nodes), model.p[i, t] - model.DEM[i, t])
        self.model.demand_covered_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, rule=demand_covered_rule)

        # 2. Production must cover losses (Kirchoff 2)
        def demand_covered_rule2(model, i, t):
            # \sum_i e_ijt = p_it - DEM_it
            # return sum(model.e[i, j, t] for i in model.Nodes) == model.p[j, t] - model.DEM[j, t]
            # \sum_i e_ijt ≤ p_it - DEM_it   # Poweflow que sale debe ser menor a lo que se produce menos lo que se consume 
            return pyo.inequality(sum(model.e[i,j,t] + model.loss[i,j,t] for j in model.Nodes), model.p[i, t])
        self.model.demand_covered2_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, rule=demand_covered_rule2)

        # 3. Power in line = power flow + loss
        def power_flow_rule(model, i, j, t):
            # \sum_i e_ijt + loss_ijt = V_ijt * I_ijt
            return model.e[i, j, t] + model.loss[i, j, t] == model.V[i, j] * model.intensity[i, j, t]
        self.model.power_flow_constraint = pyo.Constraint(self.model.Nodes, self.model.Nodes, self.model.Time, rule=power_flow_rule)

        # 4a. PowerFlow sign depends on direction
        def power_flow_direction_rule_e(model, i, j, t):
            # e_ijt = -(e_ijt)
            if model.CONN[i, j] == 1:
                return model.e[i, j, t] == -model.e[j, i, t]
            else:
                return model.e[i, j, t] == 0
        self.model.power_flow_direction_e_constraint = pyo.Constraint(self.model.Nodes, self.model.Nodes, self.model.Time, rule=power_flow_direction_rule_e)
        
        # 4b. PowerFlow sign depends on direction
        def power_flow_direction_rule_loss(model, i, j, t):
            # loss_ijt = -(loss_ijt)
            if model.CONN[i, j] == 1:
                return model.loss[i, j, t] == -model.loss[j, i, t]
            else:
                return model.loss[i, j, t] == 0
        self.model.power_flow_direction_loss_constraint = pyo.Constraint(self.model.Nodes, self.model.Nodes, self.model.Time, rule=power_flow_direction_rule_loss)

        # 5. Loss definition
        def loss_definition_rule(model, i, j, t):
            # loss_ijt = R_ij * L_ij * I_ij 
            if model.CONN[i, j] == 1:
                return model.loss[i, j, t] == model.R[i, j] * model.L[i, j] * model.intensity[i, j, t]
            else:
                return pyo.Constraint.Skip
        self.model.loss_definition_constraint = pyo.Constraint(self.model.Nodes, self.model.Nodes, self.model.Time, rule=loss_definition_rule)

        # 6. PowerFlow limits (Assuming MAXPOWERFLOW is given in the parameters)
        def power_flow_limits_rule(model, i, j, t):
            # -MAXPOWERFLOW < e_ijt < MAXPOWERFLOW
            return pyo.inequality(-model.MAXPOWERFLOW[i, j], model.e[i, j, t], model.MAXPOWERFLOW[i, j])
        self.model.power_flow_limits_constraint = pyo.Constraint(self.model.Nodes, self.model.Nodes, self.model.Time, rule=power_flow_limits_rule)

        # 7a. No PowerFlow between non-connected nodes:
        def no_power_flow_rule_R(model, i, j, t):
            if model.CONN[i, j] == 0:
                return model.R[i, j] == 0
            else:
                return pyo.Constraint.Skip

        # 7b. No PowerFlow between non-connected nodes:
        def no_power_flow_rule_V(model, i, j, t):
            if model.CONN[i, j] == 0:
                return model.V[i, j] == 0
            else:
                return pyo.Constraint.Skip

        
        # 8. Positive production
        def positive_production_rule(model, i, t):
            # p_it >= 0
            return model.p[i, t] >= sum(model.W[i, p] * model.MINPROD[p] for p in model.PrTy)
        self.model.positive_production_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, rule=positive_production_rule)

        # 9. Define maximum productions
        def max_production_rule(model, i, t):
            # p_it < W_ip MAXPROD_p
            return model.p[i, t] <= sum(model.W[i, p] * model.MAXPROD[p] for p in model.PrTy)
        self.model.max_production_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, rule=max_production_rule)



        # 10 Rules for turn on / off of normal production nodes

        # 10.1  p_it ≤ MAXPROD_p * a_it
        def r1(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                return pyo.inequality(model.p[i,t], model.MAXPROD[p]*model.a[i,t])
            else:
                return pyo.Constraint.Skip
        self.model.r1_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r1)

        # 10.2 p_it ≥ THRESHOLD_p * a_it
        def r2(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                return pyo.inequality(model.THRESHOLD[p]*model.a[i,t], model.p[i,t])
            else:
                return pyo.Constraint.Skip
        self.model.r2_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r2)

        # 10.3 a_{i,t-1} ≥ a_it + ε -> off = 1   ~   a_{i,t-1} ≤ a_it + ε + M * off_it
        def r3a(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                if t > min(self.model.Time):
                    M = 2
                    m = -M
                    use_epsilon = 1
                    ep = M / 1000 * use_epsilon
                    return pyo.inequality(model.a[i,t-1], (model.a[i,t]) + ep + M*model.off[i,t])
                else:
                    return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip
        self.model.r3a_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r3a)

        # 10.4 off = 1 -> a_{i,t-1} ≥ a_it + ε   ~   a_{i,t-1} ≥ a_it + ε + m * (1-off_it)
        def r3b(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                if t > min(self.model.Time):
                    M = 2
                    m = -M
                    use_epsilon = 1
                    ep = M / 1000 * use_epsilon
                    return pyo.inequality(model.a[i,t] + ep + m*(1-model.off[i,t]), model.a[i,t-1])
                else:
                    return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip
        self.model.r3b_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r3b)

        # 10.5 a_it ≥ a_{i,t-1} + ε -> on = 1   ~   a_it ≤ a_{i,t-1} + ε + M * on_it
        def r4a(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                if t > min(self.model.Time):
                    M = 2
                    m = -M
                    use_epsilon = 1
                    ep = M / 1000 * use_epsilon
                    return pyo.inequality(model.a[i,t], (model.a[i,t-1]) + ep + M*model.on[i,t])
                else:
                    return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip
        self.model.r4a_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r4a)

        # 10.6 on = 1 -> a_it ≥ a_{i,t-1} + ε   ~   a_it ≥ a_{i,t-1} + ε + m * (1-on_it)
        def r4b(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.turn_off_turn_on_rules_affected_types] and model.W[i,p] == 1:
                if t > min(self.model.Time):
                    M = 2
                    m = -M
                    use_epsilon = 1
                    ep = M / 1000 * use_epsilon
                    return pyo.inequality(model.a[i,t-1] + ep + m*(1-model.on[i,t]), model.a[i,t])
                else:
                    return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip
        self.model.r4b_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=r4b)



        # TRICK FOR MAKING RELAXATIONS
        # M = max(max(a), max(b))
        # m = -m

        # 1) Implication on Delta: a ≥ b -> δ = x {0/1}

        #   1.1) Reverse the inequality: a ≤ b
        #   1.2) Add M*f(δ) to the larger side: a ≤ b + M*f(δ)
        #        (or m*f(δ) to the smaller side): a + m*f(δ) ≤ b
        #   1.3) Force a ≤ b + M*f(δ) to always hold [Determining if f(δ) = δ or f(δ) = (1-δ)] such that:
        #           If a ≤ b = True  -> M*f(δ=x) is unrestricted (it will hold whether f(δ=x) = 0 or f(δ=x) = 1)
        #           If a ≤ b = False -> f(δ=x) = M (f(δ=x) = 1 is the only way for the inequality a ≤ b + M*f(δ) to hold)

        # Note. The inequality is reversed because implications can only be made when: a ≤ b = False (since if [a ≤ b = True] and the inequality doesn't always hold, unreachable scenarios will arise)

        # 2) Implication from Delta: δ = x {0/1} -> a ≤ b
        #   2.1) Add M*f(δ) to the larger side: a ≤ b + M*f(δ)
        #        (or m*f(δ) to the smaller side): a + m*f(δ) ≤ b
        #   2.2) Relax the restriction when: δ = ¬x
        #        such that:
        #           a ≤ b + M*f(δ=x)  ~ a ≤ b
        #           a ≤ b + M*f(δ=¬x) ~ a ≤ ∞ ~ a ≤ b + M



        # 10. Control cosntraints for special types of production
        def control_rule_1(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.special_types] and model.W[i,p] == 1:
                if p == self.production_type_indexes['none']:
                    return model.a[i,t] == 0
                else:
                    return model.a[i,t] == 1
            else:
                return pyo.Constraint.Skip
        self.model.crl_rule1_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=control_rule_1)

        def control_rule_2(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.special_types] and model.W[i,p] == 1:
                return model.on[i,t] == 0
            else:
                return pyo.Constraint.Skip
        self.model.crl_rule2_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=control_rule_2)

        def control_rule_3(model, i, t, p):
            if p in [self.production_type_indexes[production_type] for production_type in self.special_types] and model.W[i,p] == 1:
                return model.off[i,t] == 0
            else:
                return pyo.Constraint.Skip
        self.model.crl_rule3_constraint = pyo.Constraint(self.model.Nodes, self.model.Time, self.model.PrTy, rule=control_rule_3)





        # 11. Hydraulic production matches daily stipulated
        def hydraulic_production_rule(model, i, p):
            if p == self.production_type_indexes['hydraulic'] and model.W[i,p]:
                # print('W_it', model.W[i,p])
                return sum(model.p[i, t] for t in model.Time) == model.dailyHyd[i]
            else:
                return pyo.Constraint.Skip
        self.model.hydraulic_production_constraint = pyo.Constraint(self.model.Nodes, self.model.PrTy, rule=hydraulic_production_rule)

        # 12. Daily stipulated
        def daily_stipulated_rule(model, i):
            return model.dailyHyd[i] == model.PH[i] * model.H[i]
        self.model.daily_stipulated_constraint = pyo.Constraint(self.model.Nodes, rule=daily_stipulated_rule)

        # 13. Solar productions are fixed
        def solar_production_rule(model, i, p, t):
            if p == self.production_type_indexes['solar'] and model.W[i,p]:
                return model.p[i, t] == model.SOLAR[i, t]
            else:
                return pyo.Constraint.Skip
        self.model.solar_production_constraint = pyo.Constraint(self.model.Nodes, self.model.PrTy, self.model.Time, rule=solar_production_rule)

        # 14. Eolic productions are fixed
        def eolic_production_rule(model, i, p, t):
            if p == self.production_type_indexes['eolic'] and model.W[i,p]:
                return model.p[i, t] == model.WIND[i, t]
            else:
                return pyo.Constraint.Skip
        self.model.eolic_production_constraint = pyo.Constraint(self.model.Nodes, self.model.PrTy, self.model.Time, rule=eolic_production_rule)


    def define_solver_path(self, solver_name:str, solver_path:str = None):

        if solver_path == None:
            self.solver = pyo.SolverFactory(solver_name)
        else:
            self.solver = pyo.SolverFactory(solver_name, executable=solver_path)


    def optimize_problem(self):

        # Define model structure
        self.define_sets()
        self.define_params()
        self.define_variables()
        self.define_objective()
        self.define_constraints()
        if self.show_constraints:
            print('DISPLAY CONSTRAINTS:')
            self.display_constraints()
            print('\n'*3)

        # Optimize
        # self.model.write("model.lp", format="lp")
        # results = self.solver.solve(self.model, tee=True, options={"cmd_options": "--nopresol"})
        self.results = self.solver.solve(self.model, tee=True)  # tee=True will print solver output
        if self.results.solver.status == pyo.SolverStatus.ok and self.results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Successful optimization
            self.obj_value = pyo.value(self.model.objective)
            # Extract other relevant solution details if needed
            return True, self.obj_value
        elif self.results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            return False, 'Infeasible'
        else:
            # Something else is wrong
            return True, str(self.results.solver.status) + " - " + str(self.results.solver.termination_condition)


    def show_results(self):

        def get_value(var):
            try:
                return pyo.value(var) if not var.is_fixed() else 'NA'
            except:
                return '-'

        p_values = {(i, t): get_value(self.model.p[i, t]) for i in self.model.Nodes for t in self.model.Time}
        a_values = {(i, t): get_value(self.model.a[i, t]) for i in self.model.Nodes for t in self.model.Time}
        on_values = {(i, t): get_value(self.model.on[i, t]) for i in self.model.Nodes for t in self.model.Time}
        off_values = {(i, t): get_value(self.model.off[i, t]) for i in self.model.Nodes for t in self.model.Time}
        dly_values = {i: get_value(self.model.dailyHyd[i]) for i in self.model.Nodes}
        e_values = {(i, j, t): get_value(self.model.e[i, j, t]) for i in self.model.Nodes for j in self.model.Nodes for t in self.model.Time}

        print('\n\nVARIABLE AND OBJECTIVE RESULTS:')
        # shadow_prices = {c: pyo.value(self.model.dual[c]) for c in self.model.component_objects(pyo.Constraint, active=True)}
        print('Production Per Node (kwh):')
        for i in range(len(self.model.Nodes)):
            node_data = f'Node {i+1})'
            for t in range(len(self.model.Time)):
                p_val = pyo.value(p_values[(i+1,t+1)])
                if not isinstance(p_val, (int, float)):
                    print(f"Unexpected value for p[{i+1},{t+1}]: {p_val}")
                else:
                    node_data += f' {t}:{int(np.round(p_val,0))}'
            print(node_data)


        print('\nActive Production Node (Binary):')
        for i in range(len(self.model.Nodes)):
            node_data = f'Node {i+1})'
            for t in range(len(self.model.Time)):
                a_val = pyo.value(a_values[(i+1,t+1)])
                if not isinstance(a_val, (int, float)):
                    print(f"Unexpected value for a[{i+1},{t+1}]: {a_val}")
                else:
                    node_data += f' {t}:{int(np.round(a_val,0))}'
            print(node_data)


        print('\nON OFF (Binary):')
        for i in range(len(self.model.Nodes)):
            node_data = f'Node {i+1})'
            for t in range(len(self.model.Time)):
                on = pyo.value(on_values[(i+1,t+1)])
                off = pyo.value(off_values[(i+1,t+1)])
                node_data += f' {t}:{int(on)},{int(off)}'
            print(node_data)


        print('\ndailyHyd (kwh):')
        for i in range(len(self.model.Nodes)):
            node_data = f'Node {i+1})'
            d_val = pyo.value(dly_values[i+1])
            if not isinstance(d_val, (int, float)):
                print(f"Unexpected value for dailyHyd[{i+1}]: {d_val}")
            else:
                node_data += f' {int(np.round(d_val,0))}'
            print(node_data)


        print('\nConsumption Per Node (kwh):')
        for i in range(len(self.model.Nodes)):
            node_data = f'Node {i+1})'
            for t in range(len(self.model.Time)):
                node_data += f' {t}:{int(np.round(self.model.DEM[i+1, t+1],1))}'
            print(node_data)


        print('\n\nObjetive function OPTIMAL value:')
        print('value:', np.round(self.obj_value,2), '€')
        


        print('\nSolver Status:')
        print(self.results.solver.status)

        print('\nTermination Condition:')
        print(self.results.solver.termination_condition)


        data = {'p':p_values, 'a':a_values, 'on':on_values, 'off':off_values, 'dly':dly_values, 'e':e_values}
        return data


    def display_constraints(self):
        for constraint in self.model.component_objects(pyo.Constraint, active=True):
            print("Constraint:", constraint.name)
            for index in constraint:
                print("  ", index, ":", constraint[index].expr)
