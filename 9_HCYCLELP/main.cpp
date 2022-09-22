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
	
	hcycle(allo,K);
	allo.printInfo(pathAndFileout);
}


int hcycle(Allocation& allo, const int& K){

	// Local variables
	vector<vector<int> > hcycles;
	vector<double> RC;
	vector<vector<vector<vector<vector<int> > > > > khcycles(allo.nbNodes);
	for(int i = 0; i < allo.nbNodes;i++){
		khcycles[i].resize(allo.nbNodes);
		for(int j = 0; j < allo.nbNodes;j++){
			khcycles[i][j].resize(K);
		}
	}
	
	// Create Floyd's matrix
	vector<vector<int> > flop (allo.nbNodes, vector<int> (allo.nbNodes, allo.nbNodes));
	for(int i = 0; i < allo.adj.size();i++){
		for(int j = 0; j < allo.adj[i].size();j++)
			flop[i][allo.adj[i][j]] = 1; 	
	}
	
	for(int i = 0; i < allo.nbNodes;i++){
		for(int j = 0; j < allo.nbNodes;j++){
			for(int k = 0; k < allo.nbNodes;k++){
					flop[j][k] = min(flop[j][k],flop[j][i] + flop[i][k]);
			}
		}
	}
		
	// Create half cycles
	for(int i = 0; i < allo.nbNodes;i++){
		vector<vector<int> > curC;
		curC.push_back({i});
		for(int j = 1; j <= (K+1)/2; j++){
			vector<vector<int> > newC;
			for(int k = 0; k < curC.size();k++){
				for(int l = 0; l< allo.adj[curC[k].back()].size();l++){
					if(allo.adj[curC[k].back()][l] != i && flop[allo.adj[curC[k].back()][l]][i] <= K - j){
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
			for(int k = 0; k < newC.size();k++){
				bool add = true;
				for(int l = 1 ; l < newC[k].size()-1; l++){
					if(newC[k][l] < newC[k][0] && newC[k][l] < newC[k].back())
						add = false;
				}
				if(add && (K%2 == 0 || j < (K+1)/2 || newC[k][0] < newC[k].back()))
					khcycles[newC[k][0]][newC[k].back()][newC[k].size()-1].push_back(newC[k]);
			}
			curC = newC;
		}
	}		
	
	for(int i = 0; i < allo.nbNodes;i++){
		for(int j = i+1; j < allo.nbNodes;j++){
			for(int k = 1; k < K; k++){
				if(khcycles[i][j][k].size() > 0 && khcycles[j][i][k].size() + khcycles[j][i][k-1].size() > 0){
					for(int l = 0; l < khcycles[i][j][k].size();l++)
						hcycles.push_back(khcycles[i][j][k][l]);
					for(int l = 0; l < khcycles[j][i][k].size();l++)
						hcycles.push_back(khcycles[j][i][k][l]);
					if(khcycles[i][j][k-1].size() == 0){
						for(int l = 0; l < khcycles[j][i][k-1].size();l++)
							hcycles.push_back(khcycles[j][i][k-1][l]);
					}
				}
			}
		}
	}

	allo.infos.timeCPU.push_back(getCPUTime() - initTimeModelCPU);
	
	cout << hcycles.size() << endl;
	RC.resize(hcycles.size(),0.0);

	// Model
	try{
		GRBEnv env = GRBemptyenv;
		env.set(GRB_DoubleParam_MemLimit, 14);
		env.start();
		
		// Local variables
		GRBModel model = GRBModel(env);
		GRBLinExpr objFun = 0;
		vector<GRBVar> isHCycleUsed (hcycles.size());
		vector<GRBLinExpr> isNodeUsedS(allo.nbNodes);
		vector<GRBLinExpr> isNodeUsedM(allo.nbNodes);
		vector<GRBLinExpr> isNodeUsedE(allo.nbNodes);
		vector<vector<GRBLinExpr> > isPNodeUsed(allo.nbNodes);
		vector<vector<bool> > isPNodeUsedB(allo.nbNodes);
	
		// Initialization
		for (int i = 0; i < hcycles.size(); i++){
			isHCycleUsed[i] = model.addVar(0, 1, 0, GRB_CONTINUOUS);
		}

		for (int i = 0; i < allo.nbNodes; i++){
			isNodeUsedS[i] = 0;
			isNodeUsedM[i] = 0;
			isNodeUsedE[i] = 0;
			isPNodeUsed[i].resize(allo.nbNodes,0);
			isPNodeUsedB[i].resize(allo.nbNodes,false);
		}

		model.update();

		// Perform values
		for (int i = 0; i < hcycles.size(); i++){		
			isNodeUsedS[hcycles[i][0]] += isHCycleUsed[i];
			for(int j=1;j<hcycles[i].size()-1;j++){
				isNodeUsedM[hcycles[i][j]] += isHCycleUsed[i];
			}
			isNodeUsedE[hcycles[i].back()] += isHCycleUsed[i];
			isPNodeUsed[hcycles[i][0]][hcycles[i].back()] += isHCycleUsed[i];
			isPNodeUsedB[hcycles[i][0]][hcycles[i].back()] = true;
			objFun += (hcycles[i].size()-1) * isHCycleUsed[i];
		}

		// Unique assignment for patients
		for (int i = 0; i < allo.nbNodes; i++)
			model.addConstr(0.5*isNodeUsedS[i] + isNodeUsedM[i] + 0.5*isNodeUsedE[i] <= 1);

		// Pair correspondance 
		for (int i = 0; i < allo.nbNodes; i++){
			for (int j = i+1; j < allo.nbNodes; j++){
				if(isPNodeUsedB[i][j] || isPNodeUsedB[j][i])
					model.addConstr(isPNodeUsed[i][j] == isPNodeUsed[j][i]);
			}
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

		// Filling Solution
		for (int i = 0; i < hcycles.size(); i++){
			if(isHCycleUsed[i].get(GRB_DoubleAttr_X) < EPSILON)
				RC[i] = isHCycleUsed[i].get(GRB_DoubleAttr_RC);
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
			env.set(GRB_DoubleParam_MemLimit, 14);
			env.start();
			
			GRBModel model = GRBModel(env);
			GRBLinExpr objFun = 0;
			vector<GRBVar> isHCycleUsed (hcycles.size());
			vector<bool> isActivated (hcycles.size(),false);
			vector<GRBLinExpr> isNodeUsedS(allo.nbNodes,0);
			vector<GRBLinExpr> isNodeUsedM(allo.nbNodes,0);
			vector<GRBLinExpr> isNodeUsedE(allo.nbNodes,0);
			vector<vector<GRBLinExpr> > isPNodeUsed(allo.nbNodes);
			vector<vector<bool> > isPNodeUsedB(allo.nbNodes);
	
			// Initialization
			for (int i = 0; i < hcycles.size(); i++){
				if(allo.infos.contUB + RC[i] + EPSILON >= allo.infos.UB){
					isActivated[i] = true;
					isHCycleUsed[i] = model.addVar(0, 1, 0, GRB_BINARY);
				}
			}

			for (int i = 0; i < allo.nbNodes; i++){
				isPNodeUsed[i].resize(allo.nbNodes,0);
				isPNodeUsedB[i].resize(allo.nbNodes,false);
			}

			model.update();

			// Perform values
			for (int i = 0; i < hcycles.size(); i++){		
				if(isActivated[i]){
					isNodeUsedS[hcycles[i][0]] += isHCycleUsed[i];
					for(int j=1;j<hcycles[i].size()-1;j++)
						isNodeUsedM[hcycles[i][j]] += isHCycleUsed[i];
					isNodeUsedE[hcycles[i].back()] += isHCycleUsed[i];
					isPNodeUsed[hcycles[i][0]][hcycles[i].back()] += isHCycleUsed[i];
					isPNodeUsedB[hcycles[i][0]][hcycles[i].back()] = true;
					objFun += (hcycles[i].size()-1) * isHCycleUsed[i];
				}
			}

			// Unique assignment for patients
			for (int i = 0; i < allo.nbNodes; i++)
				model.addConstr(0.5*isNodeUsedS[i] + isNodeUsedM[i] + 0.5*isNodeUsedE[i] <= 1);

			// Pair correspondance 
			for (int i = 0; i < allo.nbNodes; i++){
				for (int j = i+1; j < allo.nbNodes; j++){
					if(isPNodeUsedB[i][j] || isPNodeUsedB[j][i])
						model.addConstr(isPNodeUsed[i][j] == isPNodeUsed[j][i]);
				}
			}
		
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

				// If solution found
				allo.infos.LB = allo.infos.UB;	
				allo.infos.opt = true;

				// Filling Solution
				for (int i = 0; i < hcycles.size(); i++){
					if(isActivated[i] && ceil(isHCycleUsed[i].get(GRB_DoubleAttr_X) - EPSILON) == 1){
						for(int j = 0; j < hcycles[i].size();j++)
							cout << allo.xti[hcycles[i][j]] << " ";
						cout << endl;
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
