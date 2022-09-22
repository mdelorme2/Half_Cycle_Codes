#include "Allocation.h" 

/*	*************************************************************************************
	********************************** ALLOCATION ***************************************
	************************************************************************************* */
	
bool sortpair(const pair<int,int>& p1, const pair<int,int>& p2){ return p1.second <  p2.second;}

void Allocation::load(const string& path, const string& filein){
	// Local variables 
	istringstream iss;
	istringstream tempIss;
	string parser;
	string garbage;
	string nameFile = path + filein;
	string tempString;
	int tail;
	int head;;
	char tempChar;
	
	itx.resize(10000,-1);
	nbNodes = 0;

	// File opening
	ifstream file(nameFile.c_str(), ios::in);

	// File lecture
	if (file){
		// Name of the instance is filein
		name = filein;

		// 2 first lines are garbage
		getline(file, parser); iss.str(parser); iss.clear(); 
		getline(file, parser); iss.str(parser); iss.clear(); 

		// For each donor
		for(;;){
			getline(file, parser); iss.str(parser);
			do{
				tempChar = iss.get();
			}
			while(tempChar != '}' && tempChar != '"');

			if(tempChar == '}'){
				iss.clear();
				break;
			}
			
			getline(iss, tempString, '"');
			tempIss.str(tempString);
			tempIss >> tail;
			itx[tail] = nbNodes;
			xti.push_back(tail);
			tempIss.clear(); 
			iss.clear(); 
			getline(file, parser); iss.str(parser); iss.clear(); 
			getline(file, parser); iss.str(parser); iss.clear(); 
			getline(file, parser); iss.str(parser); iss.clear(); 
			getline(file, parser); iss.str(parser); iss.clear(); 
			getline(file, parser); iss.str(parser); iss.clear(); 
			
			do{
				tempChar = iss.get();
			}
			while(tempChar != '}' && tempChar != '"');
			if(tempChar == '"'){	
				iss.clear(); 
				for(;;){
					getline(file, parser); iss.str(parser); 
					do{
						tempChar = iss.get();
					}
					while(tempChar != ']' && tempChar != '{');
	//				cout << "tempChar is " << tempChar << endl;
					if(tempChar == ']'){
						iss.clear(); 
						getline(file, parser); iss.str(parser);iss.clear(); 
						break;
					}
					iss.clear(); 
					getline(file, parser); iss.str(parser);
					getline(iss, tempString, ',');
					tempIss.str(tempString);
					tempIss >> garbage;
					tempIss >> head;
					edges.push_back({tail,head});
					tempIss.clear(); 
					iss.clear(); 
					getline(file, parser); iss.str(parser); iss.clear(); 
					getline(file, parser); iss.str(parser); 
				}
			}
			else iss.clear(); 
			nbNodes++;
		}	
		
		// Pair index / degree
		vector<pair<int,int> > ideg (nbNodes);
		for(int i =0; i < nbNodes; i++){
			ideg[i].first = xti[i];
			ideg[i].second = 0;
		}
		for(int i = 0; i < edges.size();i++) ideg[itx[edges[i][0]]].second += 1;
		sort(ideg.begin(), ideg.end(), sortpair);
		
		for(int i =0; i < nbNodes; i++){
			xti[i] = ideg[i].first;
			itx[ideg[i].first] = i;
		}
		
		// Create the adjacency list
		adj.resize(nbNodes);
		for(int i = 0; i < edges.size();i++)
			adj[itx[edges[i][0]]].push_back(itx[edges[i][1]]); 
		
		// Create the adjacency matrix
		adm.resize(nbNodes,vector<int> (nbNodes, 0));
		for(int i = 0; i < edges.size();i++)
			adm[itx[edges[i][0]]][itx[edges[i][1]]] = 1;
		
		file.close();
			
	}
	else cout << "Could not open the file " << nameFile << endl;
}

void Allocation::printProb(){
	cout << "Instance " << name << endl;
	for(int i = 0; i < nbNodes;i++){
		cout << "Pair " << xti[i] << " :";
		for(int j = 0; j < adj[i].size();j++)
			cout << " " << xti[adj[i][j]];
		cout << endl;
	}
}

void Allocation::printInfo(const string& pathAndFileout){
	string nameFile = pathAndFileout;
	std::ofstream file(nameFile.c_str(), std::ios::out | std::ios::app);
	file << name << "\t" << infos.opt << "\t";
	for (int i = 0; i < infos.timeCPU.size();i++)
		file << infos.timeCPU[i] << "\t";
	file << infos.LB << "\t" << infos.UB <<  "\t"  << infos.contUB <<  "\t" << infos.nbVar << "\t" << infos.nbCons << "\t" << infos.nbNZ << endl;
	file.close();
}