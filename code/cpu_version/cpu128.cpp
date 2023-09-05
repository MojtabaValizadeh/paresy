// https://github.com/MojtabaValizadeh/paresy

#include <set>
#include <vector>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <unordered_set>

using UINT128 = unsigned __int128;

// Adding parentheses if needed
std::string bracket(std::string s) {
    int p = 0;
    for (int i = 0; i < s.length(); i++){
        if (s[i] == '(') p++;
        else if (s[i] == ')') p--;
        else if (s[i] == '+' && p <= 0) return "(" + s + ")";
    }
    return s;
}

// Generating the final RE string recursively
std::string toString(int index, const std::set<char> &alphabet, const int *leftRightIdx, const int *startPoints) {

    if (index == -2) return "eps"; // Epsilon
    if (index < alphabet.size()) {std::string s(1, *next(alphabet.begin(), index)); return s;}
    int i = 0;
    while (index >= startPoints[i]){i++;}
    i--;

    if (i % 4 == 0){
        std::string res = toString(leftRightIdx[index << 1], alphabet, leftRightIdx, startPoints);
        if (res.length() > 1) return "(" + res + ")?";
        return res + "?";
    }

    if (i % 4 == 1) {
        std::string res = toString(leftRightIdx[index << 1], alphabet, leftRightIdx, startPoints);
        if (res.length() > 1) return "(" + res + ")*";
        return res + "*";
    }

    if (i % 4 == 2) {
        std::string left = toString(leftRightIdx[index << 1], alphabet, leftRightIdx, startPoints);
        std::string right = toString(leftRightIdx[(index << 1) + 1], alphabet, leftRightIdx, startPoints);
        return bracket(left) + bracket(right);
    }

    std::string left = toString(leftRightIdx[index << 1], alphabet, leftRightIdx, startPoints);
    std::string right = toString(leftRightIdx[(index << 1) + 1], alphabet, leftRightIdx, startPoints);
    return left + "+" + right;

}

// Shortlex ordering
struct strComparison{
    bool operator () (const std::string &str1, const std::string &str2) const {
        if (str1.length() == str2.length()) return str1 < str2;
        return str1.length() < str2.length();
    }
};

// Generating the infix of a string
std::set<std::string, strComparison> infixesOf (const std::string &word) {
    std::set<std::string, strComparison> ic;
    for (int len = 0; len <= word.length(); ++len) {
        for (int index = 0; index < word.length() - len + 1 ; ++index) {
            ic.insert(word.substr(index, len));
        }
    }
    return ic;
}

// Generating of the guide table only once for the whole enumeration process
bool generatingGuideTable(int &ICsize, int &gtColumns, UINT128 &posBits, UINT128 &negBits, UINT128 **guideTable,
                             const std::vector<std::string> &pos, const std::vector<std::string> &neg) {

    // Generating infix-closure (ic) of the input strings
    std::set<std::string, strComparison> ic = {};

    for (const std::string& word : pos) {
        std::set<std::string, strComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }
    for (const std::string& word : neg) {
        std::set<std::string, strComparison> set1 = infixesOf(word);
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

    ICsize = static_cast<int> (gt.size());
    gtColumns = static_cast<int> (gt.back().size());

    if (ICsize > 128) {
        printf("Your input needs %u bits which exceeds 128 bits ", ICsize);
        printf("(current version).\nPlease use less/shorter words and run the code again.\n");
        return false;
    }

    *guideTable = new UINT128 [ICsize * gtColumns];

    for (int i = 0; i < ICsize; ++i) {
        for (int j = 0; j < gt.at(i).size(); ++j) {
            (*guideTable)[i * gtColumns + j] = gt.at(i).at(j);
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

// Regular Expression Inference (REI)
std::string REI(const unsigned short *costFun, const unsigned short maxCost, int &REcost, unsigned long &allREs, 
                int &ICsize, const std::vector<std::string> &pos, const std::vector<std::string> &neg){

    // --------------------------
    // Generating the guide table
    // --------------------------

    int gtColumns;
    UINT128 posBits{}, negBits{}, *guideTable;
    if (!generatingGuideTable(ICsize, gtColumns, posBits, negBits, &guideTable, pos, neg)) return "see_the_error";

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int c1 = costFun[0]; // cost of a
    int c2 = costFun[1]; // cost of ?
    int c3 = costFun[2]; // cost of *
    int c4 = costFun[3]; // cost of .
    int c5 = costFun[4]; // cost of +

    const int cache_capacity = 200000000;

    std::unordered_set<UINT128> visited;
    auto *cache = new UINT128 [cache_capacity + 1];
    auto *leftRightIdx = new int [2 * (cache_capacity + 1)];

    // Index of the last free position in the LanguagelangCache
    int lastIdx{};

    // Initialisation of the alphabet
    std::set<char> alphabet;
    for (auto & word : pos) for (auto ch : word) alphabet.insert(ch);
    for (auto & word : neg) for (auto ch : word) alphabet.insert(ch);

    // -----------------------------------------
    // Checking empty, epsilon, and the alphabet
    // -----------------------------------------

    #ifndef MEASUREMENT_MODE
    printf("Cost %-2d | (A) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
            c1, allREs, lastIdx, static_cast<int>(alphabet.size()) + 2);
    #endif

    // Checking empty
    allREs++;
    if (pos.empty()) return "Empty";
    visited.insert(0); // int(CS of empty) is 0

    // Checking epsilon
    allREs++;
    if ((pos.size() == 1) && (pos.at(0).empty())) return "eps";
    visited.insert(1); // int(CS of empty) is 1

    // Checking the alphabet
    UINT128 idx = 2; // Pointing to the position of the first char of the alphabet (idx 0 is for epsilon)
    auto alphabetSize = static_cast<int> (alphabet.size());

    for (int i = 0; lastIdx < alphabetSize; ++i) {

        cache[i] = idx;

        allREs++;

        std::string s(1, *next(alphabet.begin(), i));
        if ((pos.size() == 1) && (pos.at(0) == s)) return s;

        visited.insert(idx);
        
        idx <<= 1;
        lastIdx++;
    }

    // 4 for "*", ".", "+" and "?"
    auto* startPoints = new int [(maxCost + 2) * 4]();
    startPoints[c1 * 4 + 3] = lastIdx;
    startPoints[(c1 + 1) * 4] = lastIdx;

    // ---------------------------
    // Enumeration of the next REs
    // ---------------------------

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    for (REcost = c1 + 1; REcost <= maxCost; ++REcost) {

        // Once it uses a previous cost that is not fully stored, it should continue as the last round
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
                UINT128 CS = cache[i];
                if (!(CS & 1)) {
                    CS |= 1;
                    allREs++;
                    if (onTheFly) {
                        if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                            leftRightIdx[lastIdx << 1] = i;
                            startPoints[REcost * 4 + 1] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                    } else if (visited.insert(CS).second) {
                        leftRightIdx[lastIdx << 1] = i;
                        if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                            startPoints[REcost * 4 + 1] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                        cache[lastIdx++] = CS;
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
                UINT128 CS = cache[i] | 1;
                unsigned int ix = alphabetSize + 1;
                UINT128 c = 1 << ix;
                while (ix < ICsize){
                    if (!(CS & c)){
                        unsigned int index = ix * gtColumns;
                        while (guideTable[index]){
                            if ((guideTable[index] & CS) && (guideTable[index + 1] & CS))
                            {CS |= c; break;}
                            index += 2;
                        }
                    }
                    c <<= 1; ix++;
                }
                allREs++;
                if (onTheFly) {
                    if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                        leftRightIdx[lastIdx << 1] = i;
                        startPoints[REcost * 4 + 2] = lastIdx + 1;
                        return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                    }
                } else if (visited.insert(CS).second) {
                    leftRightIdx[lastIdx << 1] = i;
                    if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                        startPoints[REcost * 4 + 2] = lastIdx + 1;
                        return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                    }
                    cache[lastIdx++] = CS;
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
                    UINT128 CS1 = 0;
                    if (left & 1) CS1 |= right;
                    if (right & 1) CS1 |= left;
                    UINT128 CS2 = CS1;
                    unsigned int ix = alphabetSize + 1;
                    UINT128 c = 1 << ix;
                    while (ix < ICsize) {
                        if (!(CS1 & c)) {
                            unsigned int index = ix * gtColumns;
                            while (guideTable[index]) {
                                if (((guideTable[index] & left)) && ((guideTable[index + 1] & right)))
                                {CS1 |= c; break;}
                                index += 2;
                            }
                        }
                        if (!(CS2 & c)) {
                            unsigned int index = ix * gtColumns;
                            while (guideTable[index]) {
                                if (((guideTable[index] & right)) && ((guideTable[index + 1] & left)))
                                {CS2 |= c; break;}
                                index += 2;
                            }
                        }
                        c <<= 1; ix++;
                    }
                    allREs++;
                    if (onTheFly) {
                        if ((CS1 & posBits) == posBits && (~CS1 & negBits) == negBits) {
                            leftRightIdx[lastIdx << 1] = l;
                            leftRightIdx[(lastIdx << 1) + 1] = r;
                            startPoints[REcost * 4 + 3] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                    } else {
                        if (visited.insert(CS1).second) {
                            leftRightIdx[lastIdx << 1] = l;
                            leftRightIdx[(lastIdx << 1) + 1] = r;
                            if ((CS1 & posBits) == posBits && (~CS1 & negBits) == negBits) {
                                startPoints[REcost * 4 + 3] = lastIdx + 1;
                                return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                            }
                            cache[lastIdx++] = CS1;
                            if (lastIdx == cache_capacity) onTheFly = true;
                        }
                    }
                    allREs++;
                    if (onTheFly) {
                        if ((CS2 & posBits) == posBits && (~CS2 & negBits) == negBits) {
                            leftRightIdx[lastIdx << 1] = r;
                            leftRightIdx[(lastIdx << 1) + 1] = l;
                            startPoints[REcost * 4 + 3] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                    } else {
                        if (visited.insert(CS2).second) {
                            leftRightIdx[lastIdx << 1] = r;
                            leftRightIdx[(lastIdx << 1) + 1] = l;
                            if ((CS2 & posBits) == posBits && (~CS2 & negBits) == negBits) {
                                startPoints[REcost * 4 + 3] = lastIdx + 1;
                                return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                            }
                            cache[lastIdx++] = CS2;
                            if (lastIdx == cache_capacity) onTheFly = true;
                        }
                    }
                }
            }
        }
        startPoints[REcost * 4 + 3] = lastIdx;

        //Or
        if (c1 + c5 < c2 && 2 * c1 <= REcost - c5) { // Where cost(eps+x) < cost(x?)
            #ifndef MEASUREMENT_MODE
                int tbc = startPoints[(REcost - c1 - c5 + 1) * 4] - startPoints[(REcost - c1 - c5) * 4];
                if (tbc) printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n",
                                REcost, allREs, lastIdx, tbc);
            #endif
            for (int r = startPoints[(REcost - c1 - c5) * 4]; r < startPoints[(REcost - c1 - c5 + 1) * 4]; ++r) {
                UINT128 CS = 1 | cache[r]; // Epsilon + right (instead of right? with a better cost)
                allREs++;
                if (onTheFly) {
                    if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                        leftRightIdx[lastIdx << 1] = -2;
                        leftRightIdx[(lastIdx << 1) + 1] = r;
                        startPoints[(REcost + 1) * 4] = lastIdx + 1;
                        return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                    }
                } else if (visited.insert(CS).second) {
                    leftRightIdx[lastIdx << 1] = -2;
                    leftRightIdx[(lastIdx << 1) + 1] = r;
                    if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                        startPoints[(REcost + 1) * 4] = lastIdx + 1;
                        return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                    }
                    cache[lastIdx++] = CS;
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
                    UINT128 CS = left | right;
                    allREs++;
                    if (onTheFly) {
                        if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                            leftRightIdx[lastIdx << 1] = l;
                            leftRightIdx[(lastIdx << 1) + 1] = r;
                            startPoints[(REcost + 1) * 4] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                    } else if (visited.insert(CS).second) {
                        leftRightIdx[lastIdx << 1] = l;
                        leftRightIdx[(lastIdx << 1) + 1] = r;
                        if ((CS & posBits) == posBits && (~CS & negBits) == negBits) {
                            startPoints[(REcost + 1) * 4] = lastIdx + 1;
                            return toString(lastIdx, alphabet, leftRightIdx, startPoints);
                        }
                        cache[lastIdx++] = CS;
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

// Reading the input file
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

        getline (textFile, line);

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
                    printf("\"%s\" is in both Pos and Neg examples", word.c_str());
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

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 8) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <c1> <c2> <c3> <c4> <c5> <maxCost>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        printf("\nFor example\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s ./input.txt 1 1 1 1 1 500\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    bool argError = false;
    for (int i = 2; i < argc; ++i) {
        if (std::atoi(argv[i]) <= 0 || std::atoi(argv[i]) > SHRT_MAX) {
            printf("Argument number %d, \"%s\", should be a positive short integer.\n", i, argv[i]);
            argError = true;
        }
    }
    if (argError) return 0;

    std::string fileName = argv[1];
    std::vector<std::string> pos, neg;
    if (!readFile(fileName, pos, neg)) return 0;
    unsigned short costFun[5];
    for (int i = 0; i < 5; i++)
        costFun[i] = atoi(argv[i + 2]);
    unsigned short maxCost = atoi(argv[7]);

    // ----------------------------------
    // Regular Expression Inference (REI)
    // ----------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    unsigned long allREs{};
    int ICsize, REcost = costFun[0];

    std::string output = REI(costFun, maxCost, REcost, allREs, ICsize, pos, neg);
    if (output == "see_the_error") return 0;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    // -------------------
    // Printing the output
    // -------------------

    printf("\nPositive: "); for (const auto& p : pos) printf("\"%s\" ", p.c_str());
    printf("\nNegative: "); for (const auto& n : neg) printf("\"%s\" ", n.c_str());
    printf("\nCost Function: \"a\"=%u, \"?\"=%u, \"*\"=%u, \".\"=%u, \"+\"=%u", 
            costFun[0], costFun[1], costFun[2], costFun[3], costFun[4]);
    printf("\nSize of IC: %u", ICsize);
    printf("\nCost of Final RE: %d", REcost);
    printf("\nNumber of All REs: %lu", allREs);
    printf("\nRunning Time: %f s", (double) duration * 0.000001);
    printf("\n\nRE: \"%s\"\n", output.c_str());

    return 0;

}
