// https://github.com/MojtabaValizadeh/paresy

#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>

using UINT128 = unsigned __int128;

std::string bracket(std::string s) {
    int p = 0;
    for (int i = 0; i < s.length(); i++){
        if (s[i] == '(') p++;
        else if (s[i] == ')') p--;
        else if (s[i] == '+' && p <= 0) return "(" + s + ")";
    }
    return s;
}

std::string toString(int index, const std::set<char> &alphabet, const int *signatures, const int *startPoints) {

    if (index == -2) return "eps"; // Epsilon
    if (index < alphabet.size()) {std::string s(1, *next(alphabet.begin(), index)); return s;}
    int i = 0;
    while (index >= startPoints[i]){i++;}
    i--;

    if (i % 4 == 0){
        std::string res = toString(signatures[index << 1], alphabet, signatures, startPoints);
        if (res.length() > 1) return "(" + res + ")?";
        return res + "?";
    }

    if (i % 4 == 1) {
        std::string res = toString(signatures[index << 1], alphabet, signatures, startPoints);
        if (res.length() > 1) return "(" + res + ")*";
        return res + "*";
    }

    if (i % 4 == 2) {
        std::string left = toString(signatures[index << 1], alphabet, signatures, startPoints);
        std::string right = toString(signatures[(index << 1) + 1], alphabet, signatures, startPoints);
        return bracket(left) + bracket(right);
    }

    std::string left = toString(signatures[index << 1], alphabet, signatures, startPoints);
    std::string right = toString(signatures[(index << 1) + 1], alphabet, signatures, startPoints);
    return left + "+" + right;

}

struct strLenComparison{
    bool operator () (const std::string &str1, const std::string &str2){
        if (str1.length() == str2.length()) return str1 < str2;
        return str1.length() < str2.length();
    }
};

std::string enumerateAndContainsCheckREc(const std::set<char> &alphabet, const unsigned short maxCost,
                                         const unsigned int rows, const unsigned int columns, const UINT128 posBits,
                                         const UINT128 negBits, const unsigned short *costFun,
                                         const UINT128 *guideTable, unsigned long &allREs, int &REcost){

    //initialisation
    const int cache_capacity = 200000000;

    int c1 = costFun[0]; // cost of a
    int c2 = costFun[1]; // cost of ?
    int c3 = costFun[2]; // cost of *
    int c4 = costFun[3]; // cost of .
    int c5 = costFun[4]; // cost of +

    std::unordered_set<UINT128> visited;

    auto *cache = new UINT128 [cache_capacity + 1];
    auto *signatures = new int [2 * (cache_capacity + 1)];

    auto* startPoints = new int [(maxCost + 2) * 4](); // 4 for "*", ".", "+" and "?"

    allREs = 2;
    int lastIdx{};
    UINT128 idx = 2;
    auto nonatomicIndex = static_cast<int> (alphabet.size());

    // Alphabet
    #ifndef MEASUREMENT_MODE
        printf("Cost %-2d | (A) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
               c1, allREs, lastIdx, static_cast<int>(alphabet.size()));
    #endif
    for (; lastIdx < nonatomicIndex; ++lastIdx) {
        UINT128 REc = idx;
        if ((REc & posBits) == posBits && (~REc & negBits) == negBits)
        {return toString(lastIdx, alphabet, signatures, startPoints);}
        visited.insert(REc);
        cache[lastIdx] = REc;
        idx <<= 1;
        allREs++;
    }
    startPoints[c1 * 4 + 3] = lastIdx;
    startPoints[(c1 + 1) * 4] = lastIdx;

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    // Enumeration
    for (REcost = c1 + 1; REcost <= maxCost; ++REcost) {

        if (onTheFly) {
            int dif = REcost - shortageCost;
            if (dif == c2 || dif == c3 || dif == c1 + c4 || dif == c1 + c5) lastRound = true;
        }

        //Question Mark
        if (REcost - c2 >= c1 && c1 + c5 >= c2){
            #ifndef MEASUREMENT_MODE
                int tbc = startPoints[(REcost - c2 + 1) * 4] - startPoints[(REcost - c2) * 4 + 2];
                if (tbc) printf("Cost %-2d | (Q) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int i = startPoints[(REcost - c2) * 4 + 2]; i < startPoints[(REcost - c2 + 1) * 4]; ++i) {
                UINT128 REc = cache[i];
                if (!(REc & 1)) {
                    REc |= 1;
                    allREs++;
                    if (onTheFly) {
                        if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                            signatures[lastIdx << 1] = i;
                            startPoints[REcost * 4 + 1] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                    } else if (visited.insert(REc).second) {
                        signatures[lastIdx << 1] = i;
                        if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                            startPoints[REcost * 4 + 1] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                        cache[lastIdx++] = REc;
                        if (lastIdx == cache_capacity) onTheFly = true;
                    }
                }
            }
        }
        startPoints[REcost * 4 + 1] = lastIdx;

        //Star
        if (REcost - c3 >= c1) {
            #ifndef MEASUREMENT_MODE
                int tbc = startPoints[(REcost - c3 + 1) * 4] - startPoints[(REcost - c3) * 4 + 2];
                if (tbc) printf("Cost %-2d | (S) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int i = startPoints[(REcost - c3) * 4 + 2]; i < startPoints[(REcost - c3 + 1) * 4]; ++i) {
                UINT128 REc = cache[i] | 1;
                unsigned int ix = nonatomicIndex;
                UINT128 c = 1 << ix;
                while (ix < rows){
                    if (!(REc & c)){
                        unsigned int index = ix * columns;
                        while (guideTable[index]){
                            if ((guideTable[index] & REc) && (guideTable[index + 1] & REc))
                            {REc |= c; break;}
                            index += 2;
                        }
                    }
                    c <<= 1; ix++;
                }
                allREs++;
                if (onTheFly) {
                    if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                        signatures[lastIdx << 1] = i;
                        startPoints[REcost * 4 + 2] = lastIdx + 1;
                        return toString(lastIdx, alphabet, signatures, startPoints);
                    }
                } else if (visited.insert(REc).second) {
                    signatures[lastIdx << 1] = i;
                    if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                        startPoints[REcost * 4 + 2] = lastIdx + 1;
                        return toString(lastIdx, alphabet, signatures, startPoints);
                    }
                    cache[lastIdx++] = REc;
                    if (lastIdx == cache_capacity) onTheFly = true;
                }
            }
        }
        startPoints[REcost * 4 + 2] = lastIdx;

        //Concat
        for (int i = c1; 2 * i <= REcost - c4; ++i) {
            #ifndef MEASUREMENT_MODE
                int ln = startPoints[(i + 1) * 4] - startPoints[i * 4];
                int rn = startPoints[(REcost - i - c4 + 1) * 4] - startPoints[(REcost - i - c4) * 4];
                int tbc = 2 * ln * rn;
                if (tbc) printf("Cost %-2d | (C) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int l = startPoints[i * 4]; l < startPoints[(i + 1) * 4]; ++l) {
                UINT128 left = cache[l];
                for (int r = startPoints[(REcost - i - c4) * 4]; r < startPoints[(REcost - i - c4 + 1) * 4]; ++r) {
                    UINT128 right = cache[r];
                    UINT128 REc1 = 0;
                    if (left & 1) REc1 |= right;
                    if (right & 1) REc1 |= left;
                    UINT128 REc2 = REc1;
                    unsigned int ix = nonatomicIndex;
                    UINT128 c = 1 << ix;
                    while (ix < rows) {
                        if (!(REc1 & c)) {
                            unsigned int index = ix * columns;
                            while (guideTable[index]) {
                                if (((guideTable[index] & left)) && ((guideTable[index + 1] & right)))
                                {REc1 |= c; break;}
                                index += 2;
                            }
                        }
                        if (!(REc2 & c)) {
                            unsigned int index = ix * columns;
                            while (guideTable[index]) {
                                if (((guideTable[index] & right)) && ((guideTable[index + 1] & left)))
                                {REc2 |= c; break;}
                                index += 2;
                            }
                        }
                        c <<= 1; ix++;
                    }
                    allREs++;
                    if (onTheFly) {
                        if ((REc1 & posBits) == posBits && (~REc1 & negBits) == negBits) {
                            signatures[lastIdx << 1] = l;
                            signatures[(lastIdx << 1) + 1] = r;
                            startPoints[REcost * 4 + 3] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                    } else {
                        if (visited.insert(REc1).second) {
                            signatures[lastIdx << 1] = l;
                            signatures[(lastIdx << 1) + 1] = r;
                            if ((REc1 & posBits) == posBits && (~REc1 & negBits) == negBits) {
                                startPoints[REcost * 4 + 3] = lastIdx + 1;
                                return toString(lastIdx, alphabet, signatures, startPoints);
                            }
                            cache[lastIdx++] = REc1;
                            if (lastIdx == cache_capacity) onTheFly = true;
                        }
                    }
                    allREs++;
                    if (onTheFly) {
                        if ((REc2 & posBits) == posBits && (~REc2 & negBits) == negBits) {
                            signatures[lastIdx << 1] = r;
                            signatures[(lastIdx << 1) + 1] = l;
                            startPoints[REcost * 4 + 3] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                    } else {
                        if (visited.insert(REc2).second) {
                            signatures[lastIdx << 1] = r;
                            signatures[(lastIdx << 1) + 1] = l;
                            if ((REc2 & posBits) == posBits && (~REc2 & negBits) == negBits) {
                                startPoints[REcost * 4 + 3] = lastIdx + 1;
                                return toString(lastIdx, alphabet, signatures, startPoints);
                            }
                            cache[lastIdx++] = REc2;
                            if (lastIdx == cache_capacity) onTheFly = true;
                        }
                    }
                }
            }
        }
        startPoints[REcost * 4 + 3] = lastIdx;

        //Or
        if (c1 + c5 < c2 && 2 * c1 <= REcost - c5) { // Where C(x + Epsilon) < C(x?)
            #ifndef MEASUREMENT_MODE
                int tbc = startPoints[(REcost - c1 - c5 + 1) * 4] - startPoints[(REcost - c1 - c5) * 4];
                if (tbc) printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int r = startPoints[(REcost - c1 - c5) * 4]; r < startPoints[(REcost - c1 - c5 + 1) * 4]; ++r) {
                UINT128 REc = 1 | cache[r]; // Epsilon + right (instead of right? with a better cost)
                allREs++;
                if (onTheFly) {
                    if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                        signatures[lastIdx << 1] = -2;
                        signatures[(lastIdx << 1) + 1] = r;
                        startPoints[(REcost + 1) * 4] = lastIdx + 1;
                        return toString(lastIdx, alphabet, signatures, startPoints);
                    }
                } else if (visited.insert(REc).second) {
                    signatures[lastIdx << 1] = -2;
                    signatures[(lastIdx << 1) + 1] = r;
                    if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                        startPoints[(REcost + 1) * 4] = lastIdx + 1;
                        return toString(lastIdx, alphabet, signatures, startPoints);
                    }
                    cache[lastIdx++] = REc;
                    if (lastIdx == cache_capacity) onTheFly = true;
                }
            }
        }
        for (int i = c1; 2 * i <= REcost - c5; ++i) {
            #ifndef MEASUREMENT_MODE
                int ln = startPoints[(i + 1) * 4] - startPoints[i * 4];
                int rn = startPoints[(REcost - i - c5 + 1) * 4] - startPoints[(REcost - i - c5) * 4];
                int tbc = ln * rn;
                if (tbc) printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int l = startPoints[i * 4]; l < startPoints[(i + 1) * 4]; ++l) {
                UINT128 left = cache[l];
                for (int r = startPoints[(REcost - i - c5) * 4]; r < startPoints[(REcost - i - c5 + 1) * 4]; ++r) {
                    UINT128 right = cache[r];
                    UINT128 REc = left | right;
                    allREs++;
                    if (onTheFly) {
                        if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                            signatures[lastIdx << 1] = l;
                            signatures[(lastIdx << 1) + 1] = r;
                            startPoints[(REcost + 1) * 4] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                    } else if (visited.insert(REc).second) {
                        signatures[lastIdx << 1] = l;
                        signatures[(lastIdx << 1) + 1] = r;
                        if ((REc & posBits) == posBits && (~REc & negBits) == negBits) {
                            startPoints[(REcost + 1) * 4] = lastIdx + 1;
                            return toString(lastIdx, alphabet, signatures, startPoints);
                        }
                        cache[lastIdx++] = REc;
                        if (lastIdx == cache_capacity) onTheFly = true;
                    }
                }
            }
        }
        startPoints[(REcost + 1) * 4] = lastIdx;

        if (lastRound) break;
        if (onTheFly && shortageCost == -1) shortageCost = REcost;
    }

    if (REcost == maxCost + 1) REcost--;

    return "not_found";
}

std::set<std::string, strLenComparison> infixesOf (const std::string &word){
    std::set<std::string, strLenComparison> ic;
    for (int len = 0; len <= word.length(); ++len) {
        for (int index = 0; index < word.length() - len + 1 ; ++index) {
            ic.insert(word.substr(index, len));
        }
    }
    return ic;
}

bool initialiseGuideTableEtc(int &GTrows, int &GTcolumns, UINT128 &posBits, UINT128 &negBits, UINT128 **guideTable,
                             const std::vector<std::string> &pos, const std::vector<std::string> &neg) {

    std::set<std::string, strLenComparison> ic = {};

    for (const std::string& word : pos) {
        std::set<std::string, strLenComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }
    for (const std::string& word : neg) {
        std::set<std::string, strLenComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }

    std::vector<std::vector<UINT128>> gt;

    for(auto& word : ic) {
        std::vector<UINT128> row;
        for (int i = 1; i < word.length(); ++i) {

            int index1 = 0;
            for (auto& w : ic) {
                if (w == word.substr(0, i)) break;
                index1++;
            }
            int index2 = 0;
            for (auto& w : ic) {
                if (w == word.substr(i)) break;
                index2++;
            }

            row.push_back((UINT128) 1 << index1);
            row.push_back((UINT128) 1 << index2);
        }

        row.push_back(0);
        gt.push_back(row);
    }

    GTrows = static_cast<int> (gt.size());
    GTcolumns = static_cast<int> (gt.back().size());

    if (GTrows > 128) {
        printf("Your input needs %u bits which exceeds 126 bits ", GTrows);
        printf("(current version).\nPlease use less/shorter words and run the code again.\n");
        return false;
    }

    *guideTable = new UINT128 [GTrows * GTcolumns];

    for (int i = 0; i < GTrows; ++i) {
        for (int j = 0; j < gt.at(i).size(); ++j) {
            (*guideTable)[i * GTcolumns + j] = gt.at(i).at(j);
        }
    }

    for (auto &p : pos) {
        size_t wordIndex = distance(ic.begin(), ic.find(p));
        posBits |= ((UINT128) 1 << wordIndex);
    }
    for (auto &n : neg) {
        size_t wordIndex = distance(ic.begin(), ic.find(n));
        negBits |= ((UINT128) 1 << wordIndex);
    }

    return true;

}

void initialiseAlphabet(std::set<char> &alphabet, const std::vector<std::string> &pos, const std::vector<std::string> &neg) {
    for (auto & word : pos) for (auto ch : word) alphabet.insert(ch);
    for (auto & word : neg) for (auto ch : word) alphabet.insert(ch);
}

bool readFile(const std::string& fileName, std::vector<std::string> &pos, std::vector<std::string> &neg) {

    std::string line;
    std::ifstream textFile (fileName);

    if (textFile.is_open()) {

        while (line != "++") {
            getline (textFile, line);
            if (textFile.eof()) {
                printf("Unable to find \"++\" for positive words");
                printf("\nPlease check the input file.\n");
                return false;
            }
        }

        getline(textFile, line);

        while (line != "--") {
            std::string word = "";
            for (auto c : line) if (c != ' ' && c != '"') word += c;
            pos.push_back(word);
            getline (textFile, line);
            if (line != "--" && textFile.eof()) {
                printf("Unable to find \"--\" for negative words");
                printf("\nPlease check the input file.\n");
                return false;
            }
        }

        while (getline (textFile, line)) {
            std::string word = "";
            for (auto c : line) if (c != ' ' && c != '"') word += c;
            for (auto &p : pos) {
                if (word == p) {
                    printf("%s is in both Pos and Neg examples", line.c_str());
                    printf("\nPlease check the input file, and remove one of those.\n");
                    return false;
                }
            }
            neg.push_back(word);
        }

        textFile.close();

        return true;

    }

    printf("Unable to open the file");
    return false;
}

int main (int argc, char *argv[]) {

    std::string fileName = argv[1];
    std::vector<std::string> pos, neg;
    if (!readFile(fileName, pos, neg)) return 0;
    unsigned short costFun[5];
    for (int i = 0; i < 5; i++) costFun[i] = atoi(argv[i + 2]);
    unsigned short maxCost = atoi(argv[7]);

    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    std::set<char> alphabet;
    unsigned long allREs = 1;
    UINT128 posBits{}, negBits{}, *guideTable;
    initialiseAlphabet(alphabet, pos, neg);
    int GTrows, GTcolumns, REcost = costFun[0];

    if (!initialiseGuideTableEtc(GTrows, GTcolumns, posBits,
                                 negBits, &guideTable, pos, neg)) return 0;

    if (pos.empty()) result = "empty";
    else if ((pos.size() == 1) && (pos.at(0).empty())) {result = ""; allREs++;}
    else result = enumerateAndContainsCheckREc(alphabet, maxCost, GTrows, GTcolumns,
                                               posBits, negBits, costFun, guideTable, allREs, REcost);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    printf("\nPositive: "); for (const auto& p : pos) printf("\"%s\" ", p.c_str());
    printf("\nNegative: "); for (const auto& n : neg) printf("\"%s\" ", n.c_str());
    printf("\nCost Function: %u, %u, %u, %u, %u", costFun[0], costFun[1], costFun[2], costFun[3], costFun[4]);
    printf("\nSize of ic(.): %u", GTrows);
    printf("\nCost of Final RE: %d", REcost);
    printf("\nNumber of All REs: %lu", allREs);
    printf("\nRunning Time: %f s", (double) duration * 0.000001);
    printf("\n\nRE: \"%s\"\n", result.c_str());

    return 0;

}
