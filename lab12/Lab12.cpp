#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>

int main(void) {
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	std::mt19937 gen(0);
  	std::normal_distribution<> d(500,200);
	const int N = 1<<20;
	std::vector<int> inputdata(N);
	std::generate(inputdata.begin(), inputdata.end(),[&d,&gen](){return std::max(0,(int)std::round(d(gen)));});
	std::vector<int> inputkeys(N);
	std::generate(inputkeys.begin(), inputkeys.end(),[&d,&gen](){return gen()%20;});

	/*You are given a vector of scores between 0 and 1000 (inputdata)
	  Copy those values into a thrust::device_vector and compute
	  the average of scores (~500), and their standard deviation (~200).
	  Make sure that you return and calculate with double values
	  instead of ints, otherwise you will get integer overflow problems.
	  Hint: for st. dev. use thrust::transform_reduce
	  where the operators take integers but return doubles
	  Calculate the number of scores above 800, 500, 200.
	  Find the minimum an maximum element. Try using thrust::minmax_element!
	  Hint: use "auto" for the return type
	  Check if there is an element with value 1234
	*/
	{
        thrust::device_vector<int> d_inputdata = inputdata;
        double avg = thrust::reduce(d_inputdata.begin(), d_inputdata.end(), 0.0)/N;
        std::cout << "average is " << avg << std::endl;
        //std deviation
        double sum = thrust::transform_reduce(d_inputdata.begin(), d_inputdata.end(),
                        [avg](int a){return ((double)a-avg)*((double)a-avg);}, 0.0, thrust::plus<double>()) ;
        std::cout << "std is " << sqrt(sum/N) << std::endl;
        int larger800 = thrust::count_if(d_inputdata.begin(), d_inputdata.end(),
                        [](int a) {return a>800;});
        int larger500 = thrust::count_if(d_inputdata.begin(), d_inputdata.end(),
                        [](int a) {return a>500;});
        int larger200 = thrust::count_if(d_inputdata.begin(), d_inputdata.end(),
                        [](int a) {return a>200;});
        std::cout << "Above 800 " << larger800 <<  " Above 500 " << larger500 <<  " Above 200 " << larger200 << std::endl;

        auto ret = thrust::minmax_element(d_inputdata.begin(), d_inputdata.end());
        std::cout << "minimum: "<< *ret.first << " maximum: "<< *ret.second << std::endl;

        auto it = thrust::find(d_inputdata.begin(), d_inputdata.end(),1234);
        if (it == d_inputdata.end()) std::cout << "not found" << std::endl;
        else std::cout <<"found in position "<< std::distance(d_inputdata.begin(), it) << std::endl;
	}

	/* You are given a vector of scores between 0 and 1000 (inputdata)
	 * and each score belongs to one of 20 groups (intputkeys).
	 * Copy both vectors into thrust::device_vectors
	 * Calculate the sum of scores of each group.
	 * Hint: use reduce_by_key, which requires sorted keys.
	 * Next you will calculate the number of scores in each group
	 * by using  , but instead of reducing
	 * scores, you reduce only values "1"
	 * Hint: use a thrust::constant_iterator for that
	 * Finally, compute the average scores of each group
	 * Hint use thrust::transform with two inputs and one output
	 */
	{
		thrust::device_vector<int> d_inputdata = inputdata;
		thrust::device_vector<int> d_intputkeys = inputkeys;
		thrust::device_vector<int> keys_output_sum(20);
  		thrust::device_vector<int> values_output_sum(20);
		thrust::sort_by_key(d_intputkeys.begin(), d_intputkeys.end(), d_inputdata.begin());

		auto new_last = thrust::reduce_by_key(
			d_intputkeys.begin(),
			d_intputkeys.end(),
			d_inputdata.begin(),
			keys_output_sum.begin(),
			values_output_sum.begin()
			);

		std::cout << "sum of scores of each group" << std::endl;
		for (size_t i = 0; i < keys_output_sum.size() ; i++)
		{
			std::cout << "group "<< keys_output_sum[i] << " : "<< values_output_sum[i] <<std::endl;
		}

		thrust::constant_iterator<unsigned int> const_iter(1);
		thrust::device_vector<int> keys_output_numbers(20);
  		thrust::device_vector<int> values_output_numbers(20);

		new_last = thrust::reduce_by_key(
			d_intputkeys.begin(),
			d_intputkeys.end(),
			const_iter,
			keys_output_numbers.begin(),
			values_output_numbers.begin()
			);

		std::cout << "number of scores of each group" << std::endl;
		for (size_t i = 0; i < keys_output_numbers.size() ; i++)
		{
			std::cout << "group "<< keys_output_numbers[i] << " : "<< values_output_numbers[i] <<std::endl;
		}

		thrust::device_vector<double> averages(20);
		thrust::transform(
			values_output_sum.begin(),
			values_output_sum.end(),
			values_output_numbers.begin(),
			averages.begin(),
			thrust::divides<double>()
			);

		std::cout << "average of each group" << std::endl;
		for (size_t i = 0; i < keys_output_numbers.size() ; i++)
		{
			std::cout << "group "<< keys_output_numbers[i] << " : "<< averages[i] <<std::endl;
		}

	}

	/* Copy both arrays to the device again, and create a separate
	 * array with an index for each score (0->N-1).
	 * Sort the scores in descending order, along with the group
	 * and index values
	 * Hint: use make_zip_iterator to zip groups and indices
	 * From the best 20 scores, select the ones that are in different
	 * groups. Hint: use unique, with a zip iterator and with all
	 * 3 arrays in a single tuple. Keep in mind, that unique expects
	 * a sorted input (by group in this case).
	 * Print the indices, groups and scores of these
	 */
	{
		thrust::device_vector<int> d_inputdata = inputdata;
		thrust::device_vector<int> d_intputkeys = inputkeys;
		thrust::device_vector<int> score_index(N);
		std::iota(score_index.begin(), score_index.end(), 0);
		thrust::sort_by_key(
			d_inputdata.begin(),
			d_inputdata.end(),
        	thrust::make_zip_iterator( make_tuple( d_intputkeys.begin(), score_index.begin() ) ),
			thrust::greater<int>());

		thrust::sort_by_key(
			d_intputkeys.begin(),
			d_intputkeys.begin()+19,
        	thrust::make_zip_iterator( make_tuple( d_inputdata.begin(), score_index.begin() ) ),
			thrust::greater<int>());


		auto new_end = thrust::unique(
			thrust::make_zip_iterator(make_tuple( d_intputkeys.begin(), d_inputdata.begin(), score_index.begin())),
			thrust::make_zip_iterator(make_tuple( d_intputkeys.begin()+19, d_inputdata.begin()+19, score_index.begin()+19)));

		int i=0;
		std::cout << "best 20 scores without repeated keys" << std::endl;
		while ( thrust::get<0>(*new_end) !=  d_intputkeys[i] &&
				thrust::get<1>(*new_end) !=  d_inputdata[i] &&
				thrust::get<2>(*new_end) !=  score_index[i] )
		{

			std::cout << "indice "<< score_index[i];
			std::cout << " score "<< d_inputdata[i];
			std::cout << " key "<< d_intputkeys[i];
			std::cout << std::endl;
			i+=1;
		}

	}

	return 0;
}
