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

const std::map<Xapian::docid, FeatureVector>&
Xapian::FeatureVectorBuilder::getVectors() {
    return feature_vectors;
}

const ClusterAssignment&
Xapian::ClusteringAlgorithm::getResult() const {
    return results;
}

double
Xapian::CosineSimilarity::similarity(const FeatureVector& v1,
        const FeatureVector& v2) const {
    double inner = 0.0;
    FeatureVector::iterator v1_iter = v1.begin(), v2_iter = v2.begin();
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
        centroids.push_back(std::map(miter->get_document()));
    }
}

void
Xapian::KMeans::assign_centroids() {
    MSetIterator miter;
    std::vector<FeatureVector>::iterator citer;
    double sim, msim = std::numeric_limits<double>::max();
    int mindex;
    results.cluster_to_docs.clear();
    for(miter = mset.begin(); miter != mset.end(); miter++) {
        for(citer = centroids.begin(); citer != centroids.end(); citer++) {
            sim = metric.similarity(*citer, builder.getVectorByDocid(*miter));
            if(sim < msim) {
                msim = sim;
                mindex = *miter;
            }
        }
        results.doc_to_cluster[*miter] = mindex;
        results.cluster_to_docs[mindex].pushback(*miter);
    }
}

void
Xapian::KMeans::compute_centroids() {
    std::vector<FeatureVector>::iterator citer;
    const std::vector<Xapian::docid>& docs;
    std::vector<Xapian::docid>::iterator viter;
    FeatureVector::iterator fiter;
    double denom;
    for(citer = centroids.begin(); citer != centroids.end(); citer++) {
        *citer.clear();
        docs = results.getDocsByCluster(*citer);
        for(viter = docs.begin(); viter != docs.end(); viter++) {
            const FeatureVector& vect = builder.getVectorByDocid(*viter);
            for(fiter = vect.begin(); fiter != vect.end(); fiter++) {
                *citer[fiter->first] += fiter->second;
            }
        }
        denom = (double)docs.size();
        for(fiter = (*citer).begin(); fiter != (*citer).end(); fiter++) {
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
