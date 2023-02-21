#include "main.h"

/*	*************************************************************************************
	*************************************  MAIN *****************************************
	************************************************************************************* */

double initTimeModelCPU;

int main(int argc, char **argv){
	
	initTimeModelCPU = getCPUTime();
	
	// local variables
	Allocation allo ;
	string filein = argv[2];
	string path = argv[1];
	string pathAndFileout = argv[3];
	int K = atoi(argv[4]);
	
	// functions
	allo.load(path,filein);
	allo.printProb();
	allo.infos.timeCPU.push_back(0);
	allo.infos.K = K;
	
	cycle(allo,K);
	allo.printInfo(pathAndFileout);
}


int cycle(Allocation& allo, const int& K){

	// Local variables
	vector<vector<int> > cycles;
	vector<double> RC;
	allo.infos.nbVar =  0;
	allo.infos.nbCons = 0;
	allo.infos.nbNZ = 0;
	
	// Create Floyd's matrix
	vector<vector<int> > flop (allo.nbNodes, vector<int> (allo.nbNodes, allo.nbNodes));
	for(int i = 0; i < allo.adj.size();i++){
		for(int j = 0; j < allo.adj[i].size();j++)
			flop[i][allo.adj[i][j]] = 1; 	
	}
	
	for(int i = 0; i < allo.nbNodes;i++){
		for(int j = 0; j < allo.nbNodes;j++){
			for(int k = 0; k < allo.nbNodes;k++)
				flop[j][k] = min(flop[j][k],flop[j][i] + flop[i][k]);
		}
	}
	
	// Create cycles
	for(int i = 0; i < allo.nbNodes;i++){
		vector<vector<int> > curC;
		curC.push_back({i});
		for(int j = 1; j <= K; j++){
			vector<vector<int> > newC;
			for(int k = 0; k < curC.size();k++){
				for(int l = 0; l< allo.adj[curC[k].back()].size();l++){
					if(allo.adj[curC[k].back()][l] >= i){
						if(allo.adj[curC[k].back()][l] == i){
							cycles.push_back({curC[k]});
							allo.infos.nbNZ += curC[k].size();
						}
						else{
							if(flop[allo.adj[curC[k].back()][l]][i] <= K - j){
								bool add = true;
								for(int m = 1; m <= j -1; m++){
									if(curC[k][m] == allo.adj[curC[k].back()][l]){
										add = false;
										break;
									}
								}
								if(add){
									newC.push_back({curC[k]});
									newC.back().push_back(allo.adj[curC[k].back()][l]);
								}
							}
						}
					}
				}
			}
			curC = newC;
		}
	}		

	allo.infos.nbVar = cycles.size();
	allo.infos.nbCons = allo.nbNodes;
	allo.infos.timeCPU.push_back(getCPUTime() - initTimeModelCPU);
	
	cout << cycles.size() << endl;
	RC.resize(cycles.size(),0.0);
	
	// LP Model
	try{
		GRBEnv env = GRBemptyenv;
		env.set(GRB_DoubleParam_MemLimit, 16);
		env.start();
		
		// Local variables
		GRBModel model = GRBModel(env);
		GRBLinExpr objFun = 0;
		vector<GRBVar> isCycleUsed (cycles.size());
		vector<GRBLinExpr> isNodeUsed(allo.nbNodes,0);

		// Initialization
		for (int i = 0; i < cycles.size(); i++){
			isCycleUsed[i] = model.addVar(0, 1, 0, GRB_CONTINUOUS);
		}

		model.update();

		// Perform values
		for (int i = 0; i < cycles.size(); i++){
			for(int j=0;j<cycles[i].size();j++){
				isNodeUsed[cycles[i][j]] += isCycleUsed[i];
			}
			objFun += cycles[i].size() * isCycleUsed[i];
		}

		// Unique assignment for patients
		for (int i = 0; i < allo.nbNodes; i++){
			model.addConstr(isNodeUsed[i] <= 1);
		}
				
		// Objective function
		model.setObjective(objFun, GRB_MAXIMIZE);
		
		// Setting of Gurobi
		model.getEnv().set(GRB_DoubleParam_TimeLimit,  3600);
		model.getEnv().set(GRB_IntParam_Method, 2);
		model.getEnv().set(GRB_IntParam_Crossover, 0); 
		model.getEnv().set(GRB_IntParam_Threads, 1);
		model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
		model.optimize();

		/*for(int i = 0; i < cycles.size();i++){
		for(int j = 0; j < cycles[i].size();j++)
			cout << allo.xti[cycles[i][j]] << " ";
		cout << endl;
		}*/

		// Filling Solution
		for (int i = 0; i < cycles.size(); i++){
			if(isCycleUsed[i].get(GRB_DoubleAttr_X) < EPSILON){
			//	cout << "Cycle " << i << isCycleUsed[i].get(GRB_DoubleAttr_RC) << endl;
				RC[i] = isCycleUsed[i].get(GRB_DoubleAttr_RC);
			}
			else
				RC[i] = 0.0;
		}
		
		allo.infos.contUB = model.get(GRB_DoubleAttr_ObjVal);
		allo.infos.UB = floor(model.get(GRB_DoubleAttr_ObjVal) + EPSILON);	
	}
	
	// Exceptions
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	
		// Filling Info
		allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
		allo.infos.opt = false;
		allo.infos.LB = 0;
		return -1;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}
	
	while(1){	
		// Local variables
		try{
			GRBEnv env = GRBemptyenv;
			env.set(GRB_DoubleParam_MemLimit, 16);
			env.start();
			
			GRBModel model = GRBModel(env);
			GRBLinExpr objFun = 0;
			vector<GRBVar> isCycleUsed (cycles.size());
			vector<bool> isActivated (cycles.size(),false);
			vector<GRBLinExpr> isNodeUsed(allo.nbNodes,0);

			// Initialization
			for (int i = 0; i < cycles.size(); i++){
				if(allo.infos.contUB + RC[i] + EPSILON >= allo.infos.UB){
					isCycleUsed[i] = model.addVar(0, 1, 0, GRB_BINARY);
					isActivated[i] = true;
				}
			}	

			model.update();

			// Perform values
			for (int i = 0; i < cycles.size(); i++){
				if(isActivated[i]){
					for(int j=0;j<cycles[i].size();j++)
						isNodeUsed[cycles[i][j]] += isCycleUsed[i];
					objFun += cycles[i].size() * isCycleUsed[i];
				}
			}

			// Unique assignment for patients
			for (int i = 0; i < allo.nbNodes; i++)
				model.addConstr(isNodeUsed[i] <= 1);
		
				
			// Objective function
			model.setObjective(objFun, GRB_MAXIMIZE);
		
			// Setting of Gurobi
			model.getEnv().set(GRB_DoubleParam_TimeLimit,  3600 - (getCPUTime() - initTimeModelCPU));
			model.getEnv().set(GRB_IntParam_Method, 2);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_MIPFocus, 1);
			model.optimize();
			
			if(ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON) == allo.infos.UB){
				// Filling Info
				allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
				allo.infos.opt = false;

				// Get Info pre preprocessing
				allo.infos.nbVar =  model.get(GRB_IntAttr_NumVars);
				allo.infos.nbCons = model.get(GRB_IntAttr_NumConstrs);
				allo.infos.nbNZ = model.get(GRB_IntAttr_NumNZs);

				// If no solution found
				if (model.get(GRB_IntAttr_SolCount) < 1){
					cout << "Failed to optimize ILP. " << endl;
					allo.infos.LB  = 0;
					return -1;
				}

				// If solution found
				allo.infos.LB = allo.infos.UB;	
				allo.infos.opt = true;

				// Filling Solution
				for (int i = 0; i < cycles.size(); i++){
					if(isActivated[i]){
						if(ceil(isCycleUsed[i].get(GRB_DoubleAttr_X) - EPSILON) == 1){
							for(int j = 0; j < cycles[i].size();j++)
								cout << allo.xti[cycles[i][j]] << " ";
							cout << endl;
						}
					}
				}
				break;
			}
			else{
				if(model.get(GRB_IntAttr_Status) == 9){
					// Filling Info
					allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
					allo.infos.opt = false;

					// Get Info pre preprocessing
					allo.infos.nbVar =  model.get(GRB_IntAttr_NumVars);
					allo.infos.nbCons = model.get(GRB_IntAttr_NumConstrs);
					allo.infos.nbNZ = model.get(GRB_IntAttr_NumNZs);
					allo.infos.LB = 0;
					return -1;
				}
				else allo.infos.UB--;
			}
		}
		
		// Exceptions
		catch (GRBException e) {
			cout << "Error code = " << e.getErrorCode() << endl;
			cout << e.getMessage() << endl;
		
			// Filling Info
			allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
			allo.infos.opt = false;
			allo.infos.LB = 0;
			return -1;
		}
		catch (...) {
			cout << "Exception during optimization" << endl;
		}
	}

	// End
	return 0;
}
