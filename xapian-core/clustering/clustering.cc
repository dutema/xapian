#include <config.h>
#include <xapian/clustering.h>

int
Xapian::ClusterAssignment::getClusterByDoc(Xapian::docid docid) const {
    std::map<Xapian::docid, int>::const_iterator i = doc_to_cluster.find(docid);
    if(i == doc_to_cluster.end()) {
        throw Xapian::InvalidArgumentError("Invalid document ID");
    }
        return i->second;
}

const std::vector<Xapian::docid>&
Xapian::ClusterAssignment::getDocsByCluster(int clusterid) const {
    std::map<int, std::vector<Xapian::docid> >::const_iterator i =
            cluster_to_docs.find(clusterid);
    if(i == cluster_to_docs.end()) {
        throw Xapian::InvalidArgumentError("Invalid cluster ID");
    }
        return i->second;
}

const Xapian::FeatureVector&
Xapian::FeatureVectorBuilder::getVectorByDocid(Xapian::docid docid) const{
    std::map<Xapian::docid, Xapian::FeatureVector>::const_iterator i =
            feature_vectors.find(docid);
    if(i == feature_vectors.end()) {
        throw Xapian::InvalidArgumentError("Invalid docid");
    }
    return i->second;
}

void
Xapian::ClusteringAlgorithm::clearClusterToDocs() {
    results.cluster_to_docs.clear();
}

void
Xapian::ClusteringAlgorithm::setClusterForDoc(Xapian::docid docid,
        int clusterid) {
    results.doc_to_cluster[docid] = clusterid;
}

void
Xapian::ClusteringAlgorithm::addDocForCluster(int clusterid,
        Xapian::docid docid) {
    results.cluster_to_docs[clusterid].push_back(docid);
}

const Xapian::ClusterAssignment&
Xapian::ClusteringAlgorithm::getResults() const {
    return results;
}

double
Xapian::CosineSimilarity::similarity(const FeatureVector& v1,
        const FeatureVector& v2) const {
    double inner = 0.0;
    FeatureVector::const_iterator v1_iter = v1.begin(), v2_iter = v2.begin();
    for(; v1_iter != v1.end(); v1_iter++) {
        while(v2_iter != v2.end() && v2_iter->first < v1_iter->first) {
            v2_iter++;
        }
        if(v2_iter == v2.end()) {
            break;
        }
        if(v2_iter->first == v1_iter->first) {
            inner += v1_iter->second * v2_iter->second;
        }
    }
    return inner;
}

void Xapian::KMeans::init_centroids() {
    MSetIterator miter = mset.begin();
    int i;
    for(i = 0; i < cluster_count; ++i, miter++) {
        centroids.push_back(builder->getVectorByDocid(*miter));
    }
}

void
Xapian::KMeans::assign_centroids() {
    MSetIterator miter;
    std::vector<FeatureVector>::iterator citer;
    double sim, msim = std::numeric_limits<double>::max();
    int mindex = -1;
    clearClusterToDocs();
    for(miter = mset.begin(); miter != mset.end(); miter++) {
        for(citer = centroids.begin(); citer != centroids.end(); citer++) {
            sim = metric->similarity(*citer, builder->getVectorByDocid(*miter));
            if(sim < msim) {
                msim = sim;
                mindex = *miter;
            }
        }
        setClusterForDoc(*miter, mindex);
        addDocForCluster(mindex, *miter);
    }
}

void
Xapian::KMeans::compute_centroids() {
    std::vector<FeatureVector>::iterator citer;
    std::vector<Xapian::docid>::const_iterator viter;
    FeatureVector::const_iterator cfiter;
    FeatureVector::iterator fiter;
    double denom;
    unsigned int i;
    for(i = 0; i < centroids.size(); ++i) {
        centroids[i].clear();
        const std::vector<Xapian::docid>& docs = results.getDocsByCluster(i);
        for(viter = docs.begin(); viter != docs.end(); viter++) {
            const FeatureVector& vect = builder->getVectorByDocid(*viter);
            for(cfiter = vect.begin(); cfiter != vect.end(); cfiter++) {
                centroids[i][cfiter->first] += cfiter->second;
            }
        }
        denom = (double)docs.size();
        for(fiter = centroids[i].begin(); fiter != centroids[i].end(); fiter++) {
            fiter->second /= denom;
        }
    }
}

void
Xapian::KMeans::cluster() {
    int iter;
    for(iter = 0; iter < max_iter; ++iter) {
        assign_centroids();
        compute_centroids();
    }
}
