#include <config.h>
#include <xapian/clustering.h>
#include <limits>

namespace Xapian {

clusterid
ClusterAssignment::get_cluster_by_doc(docid doc_id) const
{
    std::map<docid, clusterid>::const_iterator i = doc_to_cluster.find(doc_id);
    if (i == doc_to_cluster.end())
    {
        throw InvalidArgumentError("Invalid document ID");
    }
    return i->second;
}

const std::vector<docid>&
ClusterAssignment::get_docs_by_cluster(clusterid cluster_id) const
{
    std::map<clusterid, std::vector<docid> >::const_iterator i =
            cluster_to_docs.find(cluster_id);
    if (i == cluster_to_docs.end())
    {
        throw InvalidArgumentError("Invalid cluster ID");
    }
    return i->second;
}

const feature_vector&
FeatureVectorBuilder::get_vector_by_docid(docid doc_id) const
{
    std::map<docid, feature_vector>::const_iterator i =
            feature_vectors.find(doc_id);
    if (i == feature_vectors.end())
    {
        throw InvalidArgumentError("Invalid docid");
    }
    return i->second;
}

void
ClusteringAlgorithm::clear_cluster_to_docs()
{
    results.cluster_to_docs.clear();
}

void
ClusteringAlgorithm::set_cluster_for_doc(docid doc_id, clusterid cluster_id)
{
    results.doc_to_cluster[doc_id] = cluster_id;
}

void
ClusteringAlgorithm::add_doc_for_cluster(clusterid cluster_id, docid doc_id)
{
    results.cluster_to_docs[cluster_id].push_back(doc_id);
}

const ClusterAssignment&
ClusteringAlgorithm::get_results() const
{
    return results;
}

double
CosineSimilarity::similarity(const feature_vector& v1,
        const feature_vector& v2) const
{
    double inner = 0.0;
    feature_vector::const_iterator v1_iter = v1.begin(), v2_iter = v2.begin();
    for (; v1_iter != v1.end(); ++v1_iter)
    {
        while (v2_iter != v2.end() && v2_iter->first < v1_iter->first)
        {
            ++v2_iter;
        }
        if (v2_iter == v2.end())
        {
            break;
        }
        if (v2_iter->first == v1_iter->first)
        {
            inner += v1_iter->second * v2_iter->second;
        }
    }
    return inner;
}

void KMeans::init_centroids()
{
    for (clusterid i = 1; i <= cluster_count; ++i)
    {
        centroids.push_back(builder->get_vector_by_docid(*mset[i]));
    }
}

void
KMeans::assign_centroids()
{
    double sim, msim = std::numeric_limits<double>::max();
    int mindex = -1;
    clear_cluster_to_docs();
    for (MSetIterator miter = mset.begin(); miter != mset.end(); ++miter)
    {
        std::vector<feature_vector>::iterator citer;
        for (citer = centroids.begin(); citer != centroids.end(); ++citer)
        {
            sim = metric->similarity(*citer, builder->get_vector_by_docid(*miter));
            if (sim < msim)
            {
                msim = sim;
                mindex = *miter;
            }
        }
        set_cluster_for_doc(*miter, mindex);
        add_doc_for_cluster(mindex, *miter);
    }
}

void
KMeans::compute_centroids()
{
    double denom;
    for (unsigned int i = 0; i < centroids.size(); ++i)
    {
        centroids[i].clear();
        const std::vector<docid>& docs = results.get_docs_by_cluster(i);
        std::vector<docid>::const_iterator viter;
        for (viter = docs.begin(); viter != docs.end(); ++viter)
        {
            const feature_vector& vect = builder->get_vector_by_docid(*viter);
            feature_vector::const_iterator cfiter;
            for (cfiter = vect.begin(); cfiter != vect.end(); ++cfiter)
            {
                centroids[i][cfiter->first] += cfiter->second;
            }
        }
        denom = (double)docs.size();
        feature_vector::iterator fiter;
        for (fiter = centroids[i].begin(); fiter != centroids[i].end(); ++fiter)
        {
            fiter->second /= denom;
        }
    }
}

void
KMeans::cluster()
{
    for (unsigned int iter = 0; iter < max_iter; ++iter)
    {
        assign_centroids();
        compute_centroids();
    }
}

}
